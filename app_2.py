import cv2
import json
import os
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import tempfile
import yt_dlp
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
from streamlit_lottie import st_lottie
import json
# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

# Load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

@st.cache_resource
def load_vision_model(model_id="LiquidAI/LFM2-VL-450M"):
    """
    Load the vision-language model and processor for analyzing video frames.
    Cached to avoid reloading on every interaction.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

@st.cache_resource
def load_text_model(model_id="LiquidAI/LFM2-700M"):
    """
    Load the text model for chat functionality.
    Cached to avoid reloading on every interaction.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

@st.cache_resource
def load_embedding_model(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the sentence transformer model for semantic search.
    Used to find frames by meaning rather than exact keywords.
    """
    embedding_model = SentenceTransformer(model_id)
    return embedding_model

# ============================================================================
# VIDEO DOWNLOAD FUNCTION
# ============================================================================

def download_youtube_video(url, output_path=None):
    """
    Download video from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_path: Directory to save video (creates temp dir if None)
    
    Returns:
        tuple: (video_path, video_title) or (None, None) on error
    """
    if output_path is None:
        output_path = tempfile.mkdtemp()
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_title = info.get('title', 'video')
            return video_path, video_title
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None, None

# ============================================================================
# VIDEO PROCESSING FUNCTION (WITH TEMPORAL CONTEXT + EMBEDDINGS)
# ============================================================================

def process_video(video_path, model, processor, embedding_model, smart_filtering=True):
    """
    Process video frames with temporal context AND generate embeddings.
    This enables both VLM descriptions and semantic search.
    
    Args:
        video_path: Path to video file
        model: Vision-language model
        processor: Model processor
        embedding_model: Sentence transformer for embeddings
        smart_filtering: Whether to skip similar frames
    
    Returns:
        list: Frame data with descriptions and embeddings
    """
    all_frames = []
    skipped_count = 0
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps)  # Extract one frame per second
    
    st.info(f"📹 FPS: {fps:.2f}, extracting 1 frame per second with temporal context")
    if smart_filtering:
        st.info(f"🧠 Smart filtering enabled - VLM checks for changes")
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load all video frames into memory buffer
    # -------------------------------------------------------------------------
    status_text.text("📹 Loading video frames into memory...")
    all_video_frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_video_frames.append(frame)
        frame_count += 1
    
    cap.release()
    st.success(f"✓ Loaded {len(all_video_frames)} frames into memory")
    
    # -------------------------------------------------------------------------
    # STEP 2: Process frames with temporal context
    # -------------------------------------------------------------------------
    second_count = 0
    
    for i in range(0, len(all_video_frames), frame_interval):
        second_count += 1
        timestamp = i / fps
        
        # Update progress
        progress = i / len(all_video_frames)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {second_count} at {timestamp:.2f}s...")
        
        # Get temporal context frames (before, current, after)
        temporal_frames = []
        
        before_idx = max(0, i - 15)
        frame_rgb = cv2.cvtColor(all_video_frames[before_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))
        
        frame_rgb = cv2.cvtColor(all_video_frames[i], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))
        
        after_idx = min(len(all_video_frames) - 1, i + 15)
        frame_rgb = cv2.cvtColor(all_video_frames[after_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))
        
        # =====================================================================
        # ALWAYS ADD FIRST 5 FRAMES
        # =====================================================================
        if len(all_frames) < 5:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Frame BEFORE:"},
                        {"type": "image", "image": temporal_frames[0]},
                        {"type": "text", "text": "Frame CURRENT (describe this one):"},
                        {"type": "image", "image": temporal_frames[1]},
                        {"type": "text", "text": "Frame AFTER:"},
                        {"type": "image", "image": temporal_frames[2]},
                        {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers visible, main subjects. Be concise."},
                    ],
                },
            ]
            
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=80,
                do_sample=True,
                temperature=0.2
            )
            
            new_tokens = outputs[0][input_length:]
            description = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Generate embedding for this description
            embedding = embedding_model.encode(description, convert_to_numpy=True)
            
            # Save frame data WITH embedding
            all_frames.append({
                "frame_number": second_count,
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                "description": description,
                "embedding": embedding.tolist()  # Convert numpy to list for JSON
            })
            
            status_text.text(f"✓ Frame {second_count} added (building baseline)")
            
        # =====================================================================
        # FROM FRAME 6 ONWARDS: Smart filtering
        # =====================================================================
        else:
            if smart_filtering:
                previous_context = ""
                for prev_frame in all_frames[-5:]:
                    previous_context += f"F{prev_frame['frame_number']}: {prev_frame['description'][:60]}...\n"
                
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Previous 5 frames:\n{previous_context}\n\nNow look at these 3 frames:"},
                            {"type": "text", "text": "BEFORE:"},
                            {"type": "image", "image": temporal_frames[0]},
                            {"type": "text", "text": "CURRENT:"},
                            {"type": "image", "image": temporal_frames[1]},
                            {"type": "text", "text": "AFTER:"},
                            {"type": "image", "image": temporal_frames[2]},
                            {"type": "text", "text": "Is CURRENT frame DIFFERENT from previous? Answer ONLY 'YES' or 'NO'."},
                        ],
                    },
                ]
                
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)
                
                input_length = inputs['input_ids'].shape[1]
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.2
                )
                
                new_tokens = outputs[0][input_length:]
                response = processor.decode(new_tokens, skip_special_tokens=True).strip()
                
                response_upper = response.upper()
                is_different = "YES" in response_upper[:100] or "NEW" in response_upper[:100] or "DIFFERENT" in response_upper[:100]
                
                if is_different:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Frame BEFORE:"},
                                {"type": "image", "image": temporal_frames[0]},
                                {"type": "text", "text": "Frame CURRENT (describe this one):"},
                                {"type": "image", "image": temporal_frames[1]},
                                {"type": "text", "text": "Frame AFTER:"},
                                {"type": "image", "image": temporal_frames[2]},
                                {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers visible, main subjects. Be concise."},
                            ],
                        },
                    ]
                    
                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(model.device)
                    
                    input_length = inputs['input_ids'].shape[1]
                    
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.2
                    )
                    
                    new_tokens = outputs[0][input_length:]
                    description = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    # Generate embedding
                    embedding = embedding_model.encode(description, convert_to_numpy=True)
                    
                    all_frames.append({
                        "frame_number": second_count,
                        "timestamp_seconds": round(timestamp, 2),
                        "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                        "description": description,
                        "embedding": embedding.tolist()
                    })
                    
                    status_text.text(f"✓ Frame {second_count} added (NEW content detected)")
                else:
                    skipped_count += 1
                    status_text.text(f"⏭️ Frame {second_count} skipped (similar to previous) - {skipped_count} skipped")
            
            else:
                # Smart filtering disabled - add all frames
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Frame BEFORE:"},
                            {"type": "image", "image": temporal_frames[0]},
                            {"type": "text", "text": "Frame CURRENT (describe this one):"},
                            {"type": "image", "image": temporal_frames[1]},
                            {"type": "text", "text": "Frame AFTER:"},
                            {"type": "image", "image": temporal_frames[2]},
                            {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers, main subjects. Be concise."},
                        ],
                    },
                ]
                
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)
                
                input_length = inputs['input_ids'].shape[1]
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.2
                )
                
                new_tokens = outputs[0][input_length:]
                description = processor.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Generate embedding
                embedding = embedding_model.encode(description, convert_to_numpy=True)
                
                all_frames.append({
                    "frame_number": second_count,
                    "timestamp_seconds": round(timestamp, 2),
                    "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                    "description": description,
                    "embedding": embedding.tolist()
                })
    
    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Saved {len(all_frames)} frames, skipped {skipped_count} similar frames")
    
    return all_frames

# ============================================================================
# VIDEO CLIP EXTRACTION FUNCTION
# ============================================================================

def extract_video_clip(video_path, start_time, end_time, output_path):
    """
    Extract a video clip from start_time to end_time.
    
    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Where to save the clip
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure even dimensions
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        # Use H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate frame numbers
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)

        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        st.error(f"Error extracting clip: {e}")
        return False

# ============================================================================
# COSINE SIMILARITY FOR EMBEDDINGS
# ============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================================
# EMBEDDING-BASED SEARCH (SEMANTIC SEARCH)
# ============================================================================

def search_by_embedding(query, video_data, embedding_model, top_k=3):
    """
    Search frames using semantic similarity (embeddings).
    This finds frames by MEANING, not exact keywords.
    
    Args:
        query: User's search query
        video_data: Video frame data with embeddings
        embedding_model: Sentence transformer model
        top_k: Number of top results to return
    
    Returns:
        list: Top matching frames with similarity scores
    """
    frames = video_data['frames']
    
    # Encode the query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    # Compute similarities with all frames
    results = []
    for frame in frames:
        frame_embedding = np.array(frame['embedding'])
        similarity = cosine_similarity(query_embedding, frame_embedding)
        results.append({
            'frame_number': frame['frame_number'],
            'timestamp_seconds': frame['timestamp_seconds'],
            'timestamp_formatted': frame['timestamp_formatted'],
            'description': frame['description'],
            'similarity': float(similarity)
        })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_k]

# ============================================================================
# TEXT-BASED SEARCH WITH TIME RANGES (FLEXIBLE)
# ============================================================================

def search_by_text_flexible(user_question, video_data, text_model, tokenizer):
    """
    Search using text model with FLEXIBLE time ranges.
    Instead of exact frame, returns a time range where answer likely is.
    
    Args:
        user_question: User's search query
        video_data: Video frame data
        text_model: Text generation model
        tokenizer: Model tokenizer
    
    Returns:
        str: Response with time ranges and frame suggestions
    """
    frames = video_data['frames']
    
    # Build context from ALL frames (we need full picture for ranges)
    context = "AVAILABLE FRAMES:\n\n"
    for frame in frames:
        context += f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s): {frame['description'][:80]}...\n"
    
    # Ultra-strict prompt for time range responses
    search_prompt = f"""{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Find frames that answer the question
2. Give a TIME RANGE (not just one frame)
3. Format: "Between [time1] and [time2]" or "Around [time] (frames X-Y)"
4. If multiple possible answers, list all time ranges
5. ONLY reference frames listed above
6. If NOT FOUND, say "NOT FOUND"

Answer:"""
    
    # Generate with VERY LOW temperature for accuracy
    input_ids = tokenizer.encode(search_prompt, return_tensors="pt", truncation=True, max_length=2048).to(text_model.device)
    
    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=0.05,  # Almost deterministic
        top_p=0.9,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Validate: Check if mentioned frame numbers actually exist
    mentioned_frames = re.findall(r'[Ff]rame\s+(\d+)', response)
    valid = True
    max_frame = max([f['frame_number'] for f in frames])
    
    for frame_num in mentioned_frames:
        if int(frame_num) > max_frame:
            valid = False
            break
    
    if not valid or "NOT FOUND" in response.upper():
        return "I couldn't find frames matching your query in the available video data."
    
    return response

# ============================================================================
# UPDATED CHAT FUNCTION - 10-SECOND CLIP QA
# ============================================================================

def chat_with_video_clip(user_question, video_data, text_model, tokenizer, start_time, end_time, conversation_history=""):
    """
    Simple QA on a 10-second video clip.
    Detects if user wants facts OR generation, and adjusts accordingly.
    """
    
    # Get frames within the time window
    frames = video_data['frames']
    clip_frames = [f for f in frames if start_time <= f['timestamp_seconds'] <= end_time]
    
    if not clip_frames:
        return "No frames found in the selected time window."
    
    # =========================================================================
    # DETECT: Is this a GENERATION request or FACTUAL question?
    # =========================================================================
    question_lower = user_question.lower()
    
    generation_keywords = [
        'generate', 'create', 'make', 'similar', 'practice', 'flashcard',
        'quiz', 'questions', 'problems', 'variations', 'examples', 'like this'
    ]
    
    is_generation = any(keyword in question_lower for keyword in generation_keywords)
    
    # =========================================================================
    # GENERATION MODE - Minimal context, high temperature
    # =========================================================================
    if is_generation:
        # Only give SUMMARY, not detailed descriptions
        summary = f"VIDEO CLIP ({start_time:.1f}s - {end_time:.1f}s) SUMMARY:\n"
        summary += f"This clip shows: {clip_frames[0]['description'][:120]}...\n"
        if len(clip_frames) > 1:
            summary += f"It continues with: {clip_frames[-1]['description'][:120]}...\n"
        
        prompt = f"""{summary}

USER REQUEST: {user_question}

CRITICAL INSTRUCTIONS FOR CONTENT GENERATION:

1. ❌ DO NOT repeat or copy frame descriptions
2. ❌ DO NOT say "Frame X at..."
3. ❌ DO NOT include timestamps
4. ✅ CREATE completely NEW content inspired by the topic
5. ✅ Generate ORIGINAL questions/problems/flashcards
6. ✅ Keep the same style/difficulty as the video topic

EXAMPLE:
Bad: "Frame 1 shows 4x + 4x + 4 = 192..."
Good: "1. If 3x + 3x + 3 = 99, what is x?"

Now generate NEW content:"""
        
        temperature = 0.9  # HIGH for creativity
        max_tokens = 350
    
    # =========================================================================
    # FACTUAL MODE - Full context, low temperature
    # =========================================================================
    else:
        # Give full frame descriptions for factual questions
        context = f"VIDEO CLIP CONTENT ({start_time:.1f}s - {end_time:.1f}s):\n\n"
        for frame in clip_frames:
            context += f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s):\n{frame['description']}\n\n"
        
        if len(conversation_history) == 0:
            prompt = f"""{context}

USER QUESTION: {user_question}

RULES:
1. Answer ONLY using frame descriptions above
2. If not shown, say "I don't see that in this clip"
3. Cite frame numbers when helpful
4. Be concise and accurate

Answer:"""
            temperature = 0.1  # LOW for accuracy
        else:
            prompt = f"""{context}

PREVIOUS CONVERSATION:
{conversation_history}

USER: {user_question}

Answer:"""
            temperature = 0.15
        
        max_tokens = 250
    
    # =========================================================================
    # GENERATE RESPONSE
    # =========================================================================
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1800).to(text_model.device)
    
    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=50,  # Add diversity
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Penalize repetition
    )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # =========================================================================
    # POST-PROCESSING: Remove frame references if generation mode
    # =========================================================================
    if is_generation:
        # Clean up if model still mentions frames
        response = re.sub(r'Frame \d+ at \d+:\d+.*?:', '', response)
        response = re.sub(r'Frame \d+:', '', response)
        response = response.replace('(0.0s)', '').replace('(0.97s)', '')
        response = '\n'.join([line for line in response.split('\n') if 'timestamp' not in line.lower()])
    
    return response.strip()



# ============================================================================
# COMPLETE MAIN FUNCTION - CLEAN & MINIMAL UI
# ============================================================================
# ============================================================================
# LOTTIE ANIMATION LOADER
# ============================================================================

def load_lottiefile(filepath: str):
    """Load Lottie animation from JSON file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# ============================================================================
# COMPLETE MAIN FUNCTION - MINIMAL B&W DESIGN
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Video Frame Analyzer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for minimal B&W design
    st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f5f5f5;
}

/* REMOVE BLACK BAR - Hide header completely */
header {
    visibility: hidden;
    height: 0;
    padding: 0;
    margin: 0;
}

.main > div {
    padding-top: 1rem;
}

/* Remove extra padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* PROGRESS BAR - Dark text on light background */
.stProgress > div > div > div > div {
    background-color: #4A90E2;
}

/* Progress text - make it visible */
.stProgress {
    background-color: #e0e0e0;
}

/* Status text during processing */
.element-container div[data-testid="stMarkdownContainer"] p {
    color: #333;
}

/* Spinner text */
.stSpinner > div {
    border-top-color: #4A90E2 !important;
}

.stSpinner > div > div {
    color: #333 !important;
}

/* Clean tabs - EQUALLY SPACED */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
    padding: 0;
    border-bottom: none;
    justify-content: space-evenly;
}

.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 12px 24px;
    border: 1px solid #e0e0e0;
    color: #333;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    min-width: 200px;
}

.stTabs [aria-selected="true"] {
    background-color: #4A90E2;
    color: white;
    border: 1px solid #4A90E2;
    box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3);
}

/* File uploader styling */
.stFileUploader {
    background-color: #2d2d2d;
    border-radius: 12px;
    padding: 2rem;
    border: 2px dashed #555;
}

.stFileUploader label {
    color: #ffffff !important;
}

.stFileUploader [data-testid="stFileUploadDropzone"] {
    background-color: #2d2d2d;
}

.stFileUploader [data-testid="stFileUploadDropzone"] section {
    background-color: #2d2d2d;
    border: none;
}

.stFileUploader small {
    color: #999 !important;
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton button:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

/* Primary button */
.stButton button[kind="primary"] {
    background-color: #4A90E2;
    color: white;
    border: none;
}

/* Radio buttons */
.stRadio > label {
    font-weight: 500;
    color: #333;
}

.stRadio [role="radiogroup"] {
    gap: 1rem;
}

/* Input fields - BLACK TEXT */
.stTextInput input, .stNumberInput input {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
    color: #000000 !important;
    font-size: 1rem;
}

/* Placeholder text - gray */
.stTextInput input::placeholder {
    color: #999 !important;
}

/* Text area - BLACK TEXT */
.stTextArea textarea {
    color: #000000 !important;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

.stTextArea textarea::placeholder {
    color: #999 !important;
}

/* Select box - BLACK TEXT */
.stSelectbox div[data-baseweb="select"] > div {
    color: #000000 !important;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

.stSelectbox [data-baseweb="select"] {
    background-color: #ffffff;
}

.stSelectbox label {
    color: #333;
    font-weight: 500;
}

/* Dropdown menu items - BLACK TEXT */
[role="listbox"] [role="option"] {
    color: #000000 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    color: #2c3e50;
    font-weight: 600;
}

[data-testid="stMetricLabel"] {
    color: #666;
    font-size: 0.9rem;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    font-weight: 500;
    color: #333;
}

.streamlit-expanderContent {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Frame display */
.frame-item {
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 3px solid #4A90E2;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Chat messages */
.stChatMessage {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Video container */
.video-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #4A90E2;
    background-color: #ffffff;
}

/* Success message */
.element-container:has(.stSuccess) {
    color: #2d862d;
}

/* Info message */
.element-container:has(.stInfo) {
    color: #333;
}

/* Download button */
.stDownloadButton button {
    background-color: #2d2d2d;
    color: white;
    border: none;
    font-weight: 500;
}

.stDownloadButton button:hover {
    background-color: #1a1a1a;
}

/* Checkbox */
.stCheckbox {
    color: #333;
}

.stCheckbox label {
    color: #333 !important;
}

/* Slider */
.stSlider {
    color: #333;
}

[data-testid="stSlider"] label {
    color: #333;
}

/* Caption text */
.caption {
    color: #666 !important;
}
</style>
""", unsafe_allow_html=True)
    # Header with title on black bar
    st.markdown("""
    <div style="background-color: #1a1a1a; padding: 1.5rem 2rem; border-radius: 12px; margin: 0 auto 1rem auto; max-width: 600px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="text-align: center; color: #ffffff; margin: 0; font-size: 2rem; font-weight: 600; letter-spacing: 2px;">
            VIDEO QA
        </h1>
    </div>
""", unsafe_allow_html=True)
    
    # Lottie animation below
    lottie_animation = load_lottiefile("Animation.json")
    if lottie_animation:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st_lottie(lottie_animation, height=180, key="header_animation")
    
    # Initialize session state
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'clip_chat_history' not in st.session_state:
        st.session_state.clip_chat_history = []
    if 'current_clip_time' not in st.session_state:
        st.session_state.current_clip_time = None
    if 'video_file_path' not in st.session_state:
        st.session_state.video_file_path = None
    
    # Create tabs - ONLY 2 TABS
    tab1, tab2 = st.tabs(["📹 Analyze Video", "🔍 Search & Clips"])
    
    # =========================================================================
    # TAB 1: VIDEO ANALYSIS
    # =========================================================================
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            input_method = st.radio(
                "Choose input method:",
                ["📁 Upload File", "🔗 YouTube URL"],
                horizontal=True
            )
        
        with col2:
            smart_filtering = st.checkbox("🧠 Smart Filtering", value=True)
        
        video_path = None
        video_name = None
        
        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=['mov', 'mp4', 'avi', 'mkv', 'mpeg4'],
                help="Limit 200MB per file • MOV, MP4, AVI, MKV, MPEG4"
            )
            
            if uploaded_file is not None:
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
                video_name = uploaded_file.name
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name
                            st.session_state.video_file_path = video_path
        
        else:  # YouTube URL
            youtube_url = st.text_input("", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
            
            if youtube_url:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Download & Analyze", type="primary", use_container_width=True):
                        with st.spinner("⏳ Downloading from YouTube..."):
                            video_path, video_name = download_youtube_video(youtube_url)
                        
                        if video_path:
                            st.success(f"✓ {video_name}")
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            st.video(video_path)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.video_file_path = video_path
        
        # Process video
        if video_path:
            st.markdown("---")
            
            with st.spinner("🔄 Loading AI models..."):
                vision_model, processor = load_vision_model()
                embedding_model = load_embedding_model()
            
            results = process_video(video_path, vision_model, processor, embedding_model, smart_filtering)
            
            if results:
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                output_data = {
                    "video_name": video_name,
                    "video_path": video_path,
                    "total_frames": len(results),
                    "smart_filtering_enabled": smart_filtering,
                    "frames": results
                }
                
                st.session_state.video_data = output_data
                
                json_filename = os.path.join(output_dir, "video_analysis.json")
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                
                st.success("✅ Analysis Complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Frames", len(results))
                with col2:
                    st.metric("⏱️ Duration", f"{results[-1]['timestamp_seconds']:.0f}s")
                with col3:
                    st.metric("🎯 Quality", "Filtered" if smart_filtering else "All")
                with col4:
                    st.metric("💾 Size", f"{len(results) * 0.5:.1f}KB")
                
                # Frame preview
                st.markdown("### 📊 Frame Analysis")
                
                with st.expander(f"View all {len(results)} frames", expanded=False):
                    for frame in results:
                        st.markdown(f"""
                        <div class="frame-item">
                            <b>Frame {frame['frame_number']}</b> 
                            <span style="color: #888;">• {frame['timestamp_formatted']}</span>
                            <br>
                            <span style="font-size: 0.9rem; color: #555;">{frame['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download button
                json_str = json.dumps(output_data, indent=4)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Analysis (JSON)",
                        data=json_str,
                        file_name="video_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                st.info("✨ Navigate to **Search & Clips** tab to search and extract video clips!")
    
    # =========================================================================
    # TAB 2: SEARCH & CLIPS
    # =========================================================================
    with tab2:
        # Load existing analysis
        with st.expander("📂 Load Previous Analysis"):
            uploaded_json = st.file_uploader("Upload JSON", type=['json'], key="json_uploader_search", label_visibility="collapsed")
            if uploaded_json:
                loaded_data = json.load(uploaded_json)
                st.session_state.video_data = loaded_data
                st.success(f"✓ Loaded: {loaded_data['video_name']}")
        
        if st.session_state.video_data is None:
            st.info("👆 Please analyze a video in the **Analyze Video** tab or load an existing JSON file above")
        else:
            st.markdown(f"**📹 {st.session_state.video_data['video_name']}** • {st.session_state.video_data['total_frames']} frames")
            
            # Load models
            if 'embedding_model' not in st.session_state:
                with st.spinner("Loading search models..."):
                    st.session_state.embedding_model = load_embedding_model()
            
            if 'text_model' not in st.session_state:
                with st.spinner("Loading text models..."):
                    text_model, text_tokenizer = load_text_model()
                    st.session_state.text_model = text_model
                    st.session_state.text_tokenizer = text_tokenizer
            
            st.markdown("---")
            
            # Search interface
            col1, col2, col3 = st.columns([4, 2, 1])
            
            with col1:
                query = st.text_input("", placeholder="🔍 Search your video...", label_visibility="collapsed")
            
            with col2:
                search_method = st.selectbox("", ["Semantic", "Time Range"], label_visibility="collapsed")
            
            with col3:
                search_button = st.button("Search", type="primary", use_container_width=True)
            
            # Settings
            with st.expander("⚙️ Search Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    padding_seconds = st.slider("Clip Padding (seconds)", 0.0, 10.0, 3.0, 0.5)
                with col2:
                    top_k = st.slider("Number of Results", 1, 10, 3)
            
            if query and search_button:
                # SEMANTIC SEARCH
                if search_method == "Semantic":
                    results = search_by_embedding(
                        query, 
                        st.session_state.video_data, 
                        st.session_state.embedding_model,
                        top_k=top_k
                    )
                    
                    st.markdown(f"### 📊 Top {len(results)} Results")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Result {i}** • Frame {result['frame_number']} • {result['timestamp_formatted']}")
                            with col2:
                                st.markdown(f"**Similarity:** {result['similarity']:.2%}")
                            
                            with st.expander("📝 Description"):
                                st.write(result['description'])
                            
                            # Extract clip
                            start_time = max(0, result['timestamp_seconds'] - padding_seconds)
                            end_time = result['timestamp_seconds'] + padding_seconds
                            
                            if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
                                clips_dir = "extracted_clips"
                                os.makedirs(clips_dir, exist_ok=True)
                                
                                clip_filename = f"clip_{i}_frame{result['frame_number']}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)
                                
                                with st.spinner("Extracting clip..."):
                                    success = extract_video_clip(
                                        st.session_state.video_file_path,
                                        start_time,
                                        end_time,
                                        clip_path
                                    )
                                
                                if success:
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col2:
                                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                        st.video(clip_path)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    st.download_button(
                                        f"📥 Download Clip {i}",
                                        open(clip_path, 'rb'),
                                        file_name=clip_filename,
                                        mime="video/mp4",
                                        key=f"download_embed_{i}",
                                        use_container_width=True
                                    )
                            
                            st.markdown("---")
                
                # TEXT SEARCH
                else:
                    response = search_by_text_flexible(
                        query,
                        st.session_state.video_data,
                        st.session_state.text_model,
                        st.session_state.text_tokenizer
                    )
                    
                    st.markdown("### 📊 Search Results")
                    st.info(response)
                    
                    # Show clips for mentioned frames
                    frame_matches = re.findall(r'[Ff]rame\s+(\d+)', response)
                    
                    if frame_matches:
                        st.markdown("### 🎬 Relevant Clips")
                        
                        frames = st.session_state.video_data['frames']
                        for frame_num_str in set(frame_matches)[:3]:
                            frame_num = int(frame_num_str)
                            
                            matching_frame = next((f for f in frames if f['frame_number'] == frame_num), None)
                            
                            if matching_frame and st.session_state.video_file_path:
                                st.markdown(f"**Frame {frame_num}** • {matching_frame['timestamp_formatted']}")
                                
                                start_time = max(0, matching_frame['timestamp_seconds'] - padding_seconds)
                                end_time = matching_frame['timestamp_seconds'] + padding_seconds
                                
                                clips_dir = "extracted_clips"
                                os.makedirs(clips_dir, exist_ok=True)
                                
                                clip_filename = f"clip_frame{frame_num}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)
                                
                                success = extract_video_clip(
                                    st.session_state.video_file_path,
                                    start_time,
                                    end_time,
                                    clip_path
                                )
                                
                                if success:
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col2:
                                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                        st.video(clip_path)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    st.download_button(
                                        f"📥 Download",
                                        open(clip_path, 'rb'),
                                        file_name=clip_filename,
                                        mime="video/mp4",
                                        key=f"download_text_{frame_num}",
                                        use_container_width=True
                                    )
                                
                                st.markdown("---")


if __name__ == "__main__":
    main()