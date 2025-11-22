"""
Cloud-based video processing using E2B sandboxes, Groq VLM, and MongoDB.
Simplified version for local video files only.
"""

import os
import hashlib
import base64
import json
from typing import Dict, List, Optional, Callable
from datetime import datetime
from io import BytesIO
import certifi

from e2b_code_interpreter import Sandbox
from groq import Groq
from pymongo import MongoClient
import cv2
from PIL import Image
from pydantic import BaseModel, Field


def generate_video_id(video_path: str) -> str:
    """
    Generate a unique video ID for local video files.
    Uses SHA256 hash of file size + filename for consistency.

    Args:
        video_path: Path to video file

    Returns:
        str: Unique video ID (e.g., "video_a1b2c3d4e5f6g7h8")
    """
    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)
    hash_input = f"{filename}_{file_size}".encode()
    hash_obj = hashlib.sha256(hash_input)
    return f"video_{hash_obj.hexdigest()[:16]}"


def setup_mongodb(mongodb_uri: str, database_name: str, collection_name: str):
    """
    Setup MongoDB connection and create text search index if needed.

    Args:
        mongodb_uri: MongoDB connection string
        database_name: Database name
        collection_name: Collection name

    Returns:
        tuple: (client, collection)
    """
    # Windows-compatible TLS configuration for MongoDB Atlas
    client = MongoClient(
        mongodb_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,  # Allow self-signed certificates
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    database = client[database_name]
    collection = database[collection_name]

    # Create text search index on 'text' field for LLM agent queries
    existing_indexes = list(collection.list_indexes())
    has_text_index = any(idx.get("name") == "text_index" for idx in existing_indexes)

    if not has_text_index:
        collection.create_index([("text", "text")], name="text_index")
        print("✅ Created text search index on 'text' field")

    # Create compound index for efficient video lookups
    has_compound_index = any(idx.get("name") == "video_frame_index" for idx in existing_indexes)
    if not has_compound_index:
        collection.create_index([("video_id", 1), ("frame_number", 1)],
                                unique=True,
                                name="video_frame_index")
        print("✅ Created compound index on video_id + frame_number")

    return client, collection


def process_video_cloud(
    video_path: str,
    video_id: str,
    groq_api_key: str,
    e2b_api_key: str,
    mongodb_uri: str,
    mongodb_database: str = "video_search",
    mongodb_collection: str = "video_frames",
    sample_interval_seconds: float = 1.5,
    temporal_offset_seconds: float = 0.8,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    sandbox: Optional[Sandbox] = None
) -> Dict:
    """
    Process video using E2B sandbox with Groq VLM and MongoDB.

    Args:
        video_path: Path to video file
        video_id: Unique identifier for this video
        groq_api_key: Groq API key
        e2b_api_key: E2B API key
        mongodb_uri: MongoDB connection string
        mongodb_database: MongoDB database name
        mongodb_collection: MongoDB collection name
        sample_interval_seconds: Time between sampled frames
        temporal_offset_seconds: Time offset for before/after frames
        progress_callback: Function(current, total, status_msg) for progress updates
        sandbox: Optional existing E2B sandbox to reuse (if None, creates new one)

    Returns:
        Dictionary with processing results and sandbox instance
    """

    # Get video metadata locally
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate total samples
    sample_interval_frames = int(fps * sample_interval_seconds)
    total_samples = len(list(range(0, total_frames, sample_interval_frames)))

    # Determine if we need to create a new sandbox or reuse existing one
    should_cleanup_sandbox = False
    if sandbox is None:
        if progress_callback:
            progress_callback(0, total_samples, "Creating E2B sandbox...")

        # Clean and validate E2B API key (remove whitespace and quotes)
        clean_api_key = e2b_api_key.strip().strip('"').strip("'")

        if not clean_api_key or clean_api_key == "your_e2b_api_key_here":
            raise ValueError("Invalid E2B API key. Please set a valid API key in your .env file.")

        # Debug: Check API key format
        print(f"[DEBUG] API key length: {len(clean_api_key)}")
        print(f"[DEBUG] API key starts with: {clean_api_key[:10]}...")
        print(f"[DEBUG] API key format valid: {clean_api_key.startswith('e2b_')}")

        # Verify API key format (E2B keys should start with "e2b_")
        if not clean_api_key.startswith('e2b_'):
            raise ValueError(
                f"Invalid E2B API key format. Keys should start with 'e2b_' but got: {clean_api_key[:10]}...\n"
                "Please check your .env file. Visit https://e2b.dev/dashboard?tab=keys to get your API key."
            )

        # Create E2B sandbox using the create() method with api_key parameter
        sandbox = Sandbox.create(api_key=clean_api_key)
        should_cleanup_sandbox = True  # We created it, so we're responsible for cleanup
    else:
        if progress_callback:
            progress_callback(0, total_samples, "Reusing existing E2B sandbox...")

    try:
        if progress_callback:
            progress_callback(0, total_samples, "Uploading video to sandbox...")

        # Upload video to E2B sandbox with extended timeout for large files
        # Pass file object directly (not bytes) to allow streaming upload
        remote_video_path = f"/tmp/{os.path.basename(video_path)}"
        with open(video_path, 'rb') as f:
            # Use longer timeout for large video files (5 minutes)
            sandbox.files.write(remote_video_path, f, request_timeout=300)

        if progress_callback:
            progress_callback(0, total_samples, "Installing dependencies...")

        # Install required Python packages in sandbox
        sandbox.commands.run("pip install opencv-python-headless groq pymongo pillow dnspython")

        # Create processing script for E2B sandbox
        processing_script = f'''
import cv2
import base64
import json
from groq import Groq
from pymongo import MongoClient
from datetime import datetime
import os
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from typing import Literal

# Initialize Groq client
groq_client = Groq(api_key="{groq_api_key}")

# Initialize MongoDB client
mongo_client = MongoClient("{mongodb_uri}")
collection = mongo_client["{mongodb_database}"]["{mongodb_collection}"]

# Extract frames from video
cap = cv2.VideoCapture("{remote_video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample_interval_frames = int(fps * {sample_interval_seconds})
temporal_offset_frames = int(fps * {temporal_offset_seconds})

sampled_positions = list(range(0, total_frames, sample_interval_frames))

for idx, center_pos in enumerate(sampled_positions):
    # Check if this frame already exists in DB (skip if already processed)
    existing_frame = collection.find_one({{
        "video_id": "{video_id}",
        "frame_number": idx + 1
    }})

    if existing_frame and existing_frame.get("text") != "Error generating description":
        print(f"SKIP:{{idx + 1}}/{{len(sampled_positions)}} (already processed)")
        print(f"PROGRESS:{{idx + 1}}/{{len(sampled_positions)}}")
        continue

    # Extract 3-frame window
    before_pos = max(0, center_pos - temporal_offset_frames)
    after_pos = min(total_frames - 1, center_pos + temporal_offset_frames)

    frames = []
    for pos in [before_pos, center_pos, after_pos]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    if len(frames) != 3:
        continue

    # Convert frames to base64 for Groq API
    frame_base64_list = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        frame_b64 = base64.b64encode(buffer.getvalue()).decode()
        frame_base64_list.append(frame_b64)

    # Call Groq VLM
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {{
                    "role": "user",
                    "content": [
                        {{"type": "text", "text": "Frame BEFORE:"}},
                        {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{frame_base64_list[0]}}"}}}},
                        {{"type": "text", "text": "Frame CURRENT (describe this one):"}},
                        {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{frame_base64_list[1]}}"}}}},
                        {{"type": "text", "text": "Frame AFTER:"}},
                        {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{frame_base64_list[2]}}"}}}},
                        {{
                            "type": "text",
                            "text": "You are viewing 3 consecutive frames from a video (before, current, after). Describe what ACTION or MOVEMENT is occurring in the middle frame. Focus on: 1) What the main subjects are DOING (not just their appearance), 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."
                        }}
                    ]
                }}
            ],
            temperature=0.3,
            max_tokens=128
        )

        description = response.choices[0].message.content

    except Exception as e:
        print(f"⚠️ Groq API error for frame {{idx}}: {{e}}")
        description = "Error generating description"

    # Calculate timestamp
    timestamp = center_pos / fps

    # Create document
    document = {{
        "video_id": "{video_id}",
        "frame_number": idx + 1,
        "actual_frame_index": int(center_pos),
        "timestamp": float(timestamp),
        "text": description,
        "created_at": datetime.utcnow().isoformat()
    }}

    # Upsert to MongoDB
    collection.update_one(
        {{"video_id": document["video_id"], "frame_number": document["frame_number"]}},
        {{"$set": document}},
        upsert=True
    )

    # Report progress after every frame for live streaming
    print(f"PROGRESS:{{idx + 1}}/{{len(sampled_positions)}}")

cap.release()
print("PROCESSING_COMPLETE")
'''

        if progress_callback:
            progress_callback(0, total_samples, "Processing frames with Groq VLM...")

        # Execute processing script with streaming output and no timeout
        # Use on_stdout to process progress updates in real-time
        frames_processed = 0
        processing_complete = False
        all_stdout = []
        all_stderr = []

        def handle_stdout(line):
            nonlocal frames_processed, processing_complete
            all_stdout.append(line)
            print(f"[SANDBOX] {line}")

            if line.startswith("PROGRESS:"):
                progress_info = line.replace("PROGRESS:", "").strip()
                if '/' in progress_info:
                    try:
                        current, total = map(int, progress_info.split("/"))
                        frames_processed = current
                        if progress_callback:
                            progress_callback(current, total, f"Processing frame {current}/{total}")
                    except ValueError:
                        pass

            if "PROCESSING_COMPLETE" in line:
                processing_complete = True

        def handle_stderr(line):
            all_stderr.append(line)
            print(f"[SANDBOX ERROR] {line}")

        # Run with streaming callbacks and no timeout
        execution = sandbox.commands.run(
            f"python3 -c '{processing_script}'",
            timeout=0,
            on_stdout=handle_stdout,
            on_stderr=handle_stderr
        )

        # Check for errors
        if execution.exit_code != 0:
            error_msg = f"Sandbox script failed with exit code {execution.exit_code}\n"
            if all_stderr:
                error_msg += f"Error output:\n{chr(10).join(all_stderr)}"
            raise Exception(error_msg)

        # Verify processing completed
        if not processing_complete:
            raise Exception("Sandbox script did not complete successfully. Check output above.")

        if progress_callback:
            progress_callback(total_samples, total_samples, "✅ Processing complete!")

        return {
            "video_id": video_id,
            "total_frames": frames_processed if frames_processed > 0 else total_samples,
            "status": "complete",
            "sandbox": sandbox  # Return sandbox for reuse
        }

    except Exception as e:
        # Only cleanup sandbox if we created it
        if should_cleanup_sandbox:
            sandbox.kill()
        raise Exception(f"E2B processing error: {str(e)}")

    finally:
        # Only cleanup sandbox if we created it (not reusing)
        if should_cleanup_sandbox:
            sandbox.kill()


class SearchKeywords(BaseModel):
    """Structured output for search keyword generation."""
    keywords: list[str] = Field(description="3-5 search keywords/phrases for video frame search")
    reasoning: str = Field(description="Brief explanation of keyword choices")


class FrameResult(BaseModel):
    """Single frame search result."""
    video_id: str
    frame_number: int
    timestamp: float
    text: str
    score: float


class SearchResults(BaseModel):
    """Structured search results."""
    results: list[FrameResult] = Field(description="Top search results ranked by relevance")
    total_found: int = Field(description="Total number of matches before filtering")


def search_video_frames(
    query: str,
    groq_api_key: str,
    mongodb_collection,
    video_id: Optional[str] = None,
    top_k: int = 5,
    sandbox: Optional[Sandbox] = None
) -> List[Dict]:
    """
    Search for frames using structured LLM-powered semantic search.

    Args:
        query: Search query text
        groq_api_key: Groq API key
        mongodb_collection: PyMongo collection object
        video_id: Optional video ID to limit search to specific video
        top_k: Number of results to return
        sandbox: Optional E2B sandbox (currently unused)

    Returns:
        List of matching frame documents ranked by relevance
    """
    # Initialize Groq client
    groq_client = Groq(api_key=groq_api_key)

    # Step 1: Get sample frame descriptions for context
    sample_pipeline = [{"$sample": {"size": 5}}]
    if video_id:
        sample_pipeline.insert(0, {"$match": {"video_id": video_id}})

    sample_docs = list(mongodb_collection.aggregate(sample_pipeline))
    sample_descriptions = [doc["text"] for doc in sample_docs]

    # Step 2: Generate search keywords using structured output
    keyword_prompt = f"""Analyze this video search query and generate 3-5 relevant search keywords/phrases.

User Query: "{query}"

Example frame descriptions from database:
{chr(10).join(f'- {desc}' for desc in sample_descriptions)}

Focus on ACTION and MOVEMENT terms (e.g., "jumping", "shooting", "running", "layup", "dribbling")."""

    try:
        keyword_response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a video search keyword generator. Extract action-focused search terms."},
                {"role": "user", "content": keyword_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "search_keywords",
                    "schema": SearchKeywords.model_json_schema()
                }
            },
            temperature=0.3,
        )

        # Parse structured keywords
        keywords_data = SearchKeywords.model_validate(json.loads(keyword_response.choices[0].message.content))
        search_terms = keywords_data.keywords
        print(f"[DEBUG] Generated keywords: {search_terms}")

    except Exception as e:
        print(f"⚠️ Keyword generation error: {e}, using original query")
        search_terms = [query]

    # Step 3: Execute MongoDB text search
    search_filter = {"$text": {"$search": " ".join(search_terms)}}
    if video_id:
        search_filter["video_id"] = video_id

    results = list(mongodb_collection.find(
        search_filter,
        {"_id": 0, "score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(top_k * 3))

    if not results:
        print(f"⚠️ No results found for query: {query}")
        return []

    # Step 4: Re-rank results using structured output
    if len(results) > top_k:
        rerank_prompt = f"""Rank these video frame descriptions by relevance to the user's query.

User Query: "{query}"

Frame descriptions:
{chr(10).join(f'{i+1}. [Frame {r["frame_number"]}] {r["text"]} (timestamp: {r["timestamp"]:.1f}s, score: {r["score"]:.2f})' for i, r in enumerate(results))}

Return the top {top_k} most relevant frames with their metadata."""

        try:
            # Create Pydantic model dynamically for re-ranking
            class RankedFrame(BaseModel):
                frame_number: int
                relevance_score: float = Field(description="Semantic relevance score 0-10")

            class RerankedResults(BaseModel):
                ranked_frames: list[RankedFrame] = Field(description=f"Top {top_k} frames ranked by relevance")

            rerank_response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are a semantic video search re-ranker. Rank frames by relevance."},
                    {"role": "user", "content": rerank_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reranked_results",
                        "schema": RerankedResults.model_json_schema()
                    }
                },
                temperature=0.1,
            )

            # Parse re-ranked results
            reranked_data = RerankedResults.model_validate(json.loads(rerank_response.choices[0].message.content))

            # Map frame numbers back to original results
            frame_to_result = {r["frame_number"]: r for r in results}
            ranked_results = []
            for ranked_frame in reranked_data.ranked_frames:
                if ranked_frame.frame_number in frame_to_result:
                    result = frame_to_result[ranked_frame.frame_number].copy()
                    # Store both MongoDB score and LLM relevance score
                    result["relevance_score"] = ranked_frame.relevance_score
                    ranked_results.append(result)

            results = ranked_results[:top_k]
            print(f"[DEBUG] Re-ranked to {len(results)} results")

        except Exception as e:
            print(f"⚠️ Re-ranking error: {e}, using MongoDB text scores")
            results = results[:top_k]
    else:
        results = results[:top_k]

    return results


def _fallback_search(query: str, mongodb_collection, video_id: Optional[str], top_k: int) -> List[Dict]:
    """Fallback search using simple MongoDB text search."""
    search_filter = {"$text": {"$search": query}}
    if video_id:
        search_filter["video_id"] = video_id

    results = list(mongodb_collection.find(
        search_filter,
        {"_id": 0, "score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(top_k))

    return results
