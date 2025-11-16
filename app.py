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

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

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
# COMPLETE MAIN FUNCTION
# ============================================================================

def main():
    st.set_page_config(page_title="Video Frame Analyzer & Chat", page_icon="🎬", layout="wide")
    
    st.title("🎬 Video Frame Analyzer & Chat")
    st.markdown("Upload a video or provide a YouTube URL to analyze frames and chat about the content")
    
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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "🔍 Search & Clips", "💬 Chat with Video"])
    
    # =========================================================================
    # TAB 1: VIDEO ANALYSIS
    # =========================================================================
    with tab1:
        input_method = st.radio("Choose input method:", ["Upload File", "YouTube URL"])
        
        smart_filtering = st.checkbox("🧠 Enable Smart Filtering (VLM checks for frame changes)", value=True)
        
        video_path = None
        video_name = None
        youtube_url = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])
            
            if uploaded_file is not None:
                st.video(uploaded_file)
                video_name = uploaded_file.name
                
                if st.button("🚀 Start Analysis", type="primary", key="upload_analyze"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                        st.session_state.video_file_path = video_path
        
        else:
            youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
            
            if youtube_url:
                if st.button("🚀 Download & Analyze", type="primary", key="youtube_analyze"):
                    with st.spinner("Downloading video from YouTube..."):
                        video_path, video_name = download_youtube_video(youtube_url)
                    
                    if video_path:
                        st.success(f"✓ Downloaded: {video_name}")
                        st.video(video_path)
                        st.session_state.video_file_path = video_path
        
        if video_path:
            with st.spinner("Loading vision model..."):
                vision_model, processor = load_vision_model()
            st.success("✓ Vision model loaded!")
            
            with st.spinner("Loading embedding model..."):
                embedding_model = load_embedding_model()
            st.success("✓ Embedding model loaded!")
            
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
                
                st.success(f"✓ Saved to: {json_filename}")
                
                st.markdown("---")
                st.header("📊 Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Frames Saved", len(results))
                with col2:
                    if smart_filtering:
                        st.metric("Quality", "Filtered for changes")
                
                for frame in results:
                    col1, col2, col3 = st.columns([1, 1, 5])
                    with col1:
                        st.markdown(f"**Frame {frame['frame_number']}**")
                    with col2:
                        st.markdown(f"`{frame['timestamp_formatted']}`")
                    with col3:
                        st.markdown(frame['description'])
                
                json_str = json.dumps(output_data, indent=4)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="video_analysis.json",
                    mime="application/json"
                )
                
                st.info("✨ Go to 'Search & Clips' or 'Chat' tabs to interact with the video!")
        
        if not video_path and input_method == "Upload File":
            st.info("👆 Upload a video file to get started")
        elif not youtube_url and input_method == "YouTube URL":
            st.info("👆 Enter a YouTube URL to get started")
    
    # =========================================================================
    # TAB 2: SEARCH & VIDEO CLIPS (UNCHANGED - KEEP AS IS)
    # =========================================================================
    with tab2:
        st.markdown("### 📂 Load Video Analysis")
        uploaded_json = st.file_uploader("Upload existing JSON analysis (optional)", type=['json'], key="json_uploader_search")
        if uploaded_json:
            loaded_data = json.load(uploaded_json)
            st.session_state.video_data = loaded_data
            st.success(f"✓ Loaded: {loaded_data['video_name']} ({loaded_data['total_frames']} frames)")
        
        if st.session_state.video_data is None:
            st.warning("⚠️ Please analyze a video first or upload a JSON file")
        else:
            st.success(f"✓ Loaded: {st.session_state.video_data['video_name']}")
            st.info(f"📊 {st.session_state.video_data['total_frames']} frames available")
            
            # Load models
            if 'embedding_model' not in st.session_state:
                with st.spinner("Loading embedding model..."):
                    st.session_state.embedding_model = load_embedding_model()
                st.success("✓ Embedding model loaded!")
            
            if 'text_model' not in st.session_state:
                with st.spinner("Loading text model..."):
                    text_model, text_tokenizer = load_text_model()
                    st.session_state.text_model = text_model
                    st.session_state.text_tokenizer = text_tokenizer
                st.success("✓ Text model loaded!")
            
            # Search interface
            st.markdown("---")
            st.header("🔍 Search Video")
            
            # Search method selector
            search_method = st.radio(
                "Search method:",
                ["Embedding Search (Semantic)", "Text Search (Flexible Time Ranges)"],
                help="Embedding: Fast, finds by meaning | Text: Gives time ranges, more flexible"
            )
            
            query = st.text_input("Enter your search query:", placeholder="e.g., 'when did Wisconsin score first point?'")
            
            # Clip settings
            with st.expander("⚙️ Video Clip Settings"):
                padding_seconds = st.slider(
                    "Padding (seconds before/after)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
                top_k = st.slider("Number of results to show", 1, 10, 3)
            
            if query and st.button("🔍 Search", type="primary"):
                # EMBEDDING SEARCH
                if search_method == "Embedding Search (Semantic)":
                    results = search_by_embedding(
                        query, 
                        st.session_state.video_data, 
                        st.session_state.embedding_model,
                        top_k=top_k
                    )
                    
                    st.markdown(f"### 📊 Top {len(results)} Results (Semantic Search)")
                    
                    for i, result in enumerate(results, 1):
                        st.markdown(f"#### Result {i}: Frame {result['frame_number']} - Similarity: {result['similarity']:.3f}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"**Time:** {result['timestamp_formatted']} ({result['timestamp_seconds']}s)")
                            st.markdown(f"**Description:** {result['description']}")
                        
                        with col2:
                            start_time = max(0, result['timestamp_seconds'] - padding_seconds)
                            end_time = result['timestamp_seconds'] + padding_seconds
                            
                            st.markdown(f"**Clip Range:** {start_time:.1f}s to {end_time:.1f}s")
                            
                            if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
                                clips_dir = "extracted_clips"
                                os.makedirs(clips_dir, exist_ok=True)
                                
                                clip_filename = f"clip_{i}_frame{result['frame_number']}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)
                                
                                with st.spinner(f"Extracting clip {i}..."):
                                    success = extract_video_clip(
                                        st.session_state.video_file_path,
                                        start_time,
                                        end_time,
                                        clip_path
                                    )
                                
                                if success:
                                    st.video(clip_path)
                                    
                                    with open(clip_path, 'rb') as f:
                                        st.download_button(
                                            f"📥 Download Clip {i}",
                                            f,
                                            file_name=clip_filename,
                                            mime="video/mp4",
                                            key=f"download_embed_{i}"
                                        )
                                else:
                                    st.error("Failed to extract clip")
                            else:
                                st.warning("Video file not found. Upload video again to extract clips.")
                        
                        st.markdown("---")
                
                # TEXT SEARCH WITH FLEXIBLE TIME RANGES
                else:
                    response = search_by_text_flexible(
                        query,
                        st.session_state.video_data,
                        st.session_state.text_model,
                        st.session_state.text_tokenizer
                    )
                    
                    st.markdown("### 📊 Search Results (Text-Based with Time Ranges)")
                    st.markdown(response)
                    
                    # Extract frame numbers from response
                    frame_matches = re.findall(r'[Ff]rame\s+(\d+)', response)
                    
                    if frame_matches:
                        st.markdown("---")
                        st.markdown("### 🎬 Video Clips for Mentioned Frames:")
                        
                        frames = st.session_state.video_data['frames']
                        for frame_num_str in set(frame_matches):
                            frame_num = int(frame_num_str)
                            
                            matching_frame = next((f for f in frames if f['frame_number'] == frame_num), None)
                            
                            if matching_frame and st.session_state.video_file_path:
                                st.markdown(f"**Frame {frame_num} ({matching_frame['timestamp_formatted']})**")
                                
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
                                    st.video(clip_path)
                                    
                                    with open(clip_path, 'rb') as f:
                                        st.download_button(
                                            f"📥 Download Frame {frame_num} Clip",
                                            f,
                                            file_name=clip_filename,
                                            mime="video/mp4",
                                            key=f"download_text_{frame_num}"
                                        )
    
    # =========================================================================
    # TAB 3: 10-SECOND CLIP Q&A (NEW - UPDATED)
    # =========================================================================
    with tab3:
        st.markdown("### 📂 Load Video Analysis")
        uploaded_json_chat = st.file_uploader("Upload existing JSON analysis (optional)", type=['json'], key="json_uploader_chat")
        if uploaded_json_chat:
            loaded_data = json.load(uploaded_json_chat)
            st.session_state.video_data = loaded_data
            if 'video_path' in loaded_data and os.path.exists(loaded_data['video_path']):
                st.session_state.video_file_path = loaded_data['video_path']
            st.success(f"✓ Loaded: {loaded_data['video_name']} ({loaded_data['total_frames']} frames)")
        
        if st.session_state.video_data is None:
            st.warning("⚠️ Please analyze a video first or upload a JSON file")
        else:
            st.success(f"✓ Loaded: {st.session_state.video_data['video_name']}")
            st.info(f"📊 {st.session_state.video_data['total_frames']} frames available")
            
            if 'text_model' not in st.session_state:
                with st.spinner("Loading chat model..."):
                    text_model, text_tokenizer = load_text_model()
                    st.session_state.text_model = text_model
                    st.session_state.text_tokenizer = text_tokenizer
                st.success("✓ Chat model loaded!")
            
            st.markdown("---")
            
            # =================================================================
            # 10-SECOND CLIP SELECTION
            # =================================================================
            st.subheader("🎬 Select 10-Second Clip")
            
            frames = st.session_state.video_data['frames']
            max_time = max([f['timestamp_seconds'] for f in frames])
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                start_time = st.number_input(
                    "Start Time (seconds)",
                    min_value=0.0,
                    max_value=max_time,
                    value=0.0,
                    step=1.0,
                    format="%.1f"
                )
            
            with col2:
                clip_duration = st.number_input(
                    "Clip Duration (seconds)",
                    min_value=5.0,
                    max_value=30.0,
                    value=10.0,
                    step=1.0,
                    format="%.1f"
                )
            
            end_time = min(start_time + clip_duration, max_time)
            
            with col3:
                st.markdown("**End Time:**")
                st.markdown(f"# {end_time:.1f}s")
            
            frames_in_window = len([f for f in frames if start_time <= f['timestamp_seconds'] <= end_time])
            st.info(f"📊 This {clip_duration}s clip contains **{frames_in_window} analyzed frames**")
            
            # Quick presets
            st.markdown("**Quick Presets:**")
            preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
            
            with preset_col1:
                if st.button("⏮️ Beginning (0-10s)"):
                    st.session_state.current_clip_time = (0.0, 10.0)
                    st.session_state.clip_chat_history = []
                    st.rerun()
            
            with preset_col2:
                mid_point = max_time / 2
                if st.button(f"⏸️ Middle ({mid_point-5:.0f}-{mid_point+5:.0f}s)"):
                    st.session_state.current_clip_time = (mid_point - 5, mid_point + 5)
                    st.session_state.clip_chat_history = []
                    st.rerun()
            
            with preset_col3:
                if st.button(f"⏭️ End ({max_time-10:.0f}-{max_time:.0f}s)"):
                    st.session_state.current_clip_time = (max(0, max_time - 10), max_time)
                    st.session_state.clip_chat_history = []
                    st.rerun()
            
            with preset_col4:
                if st.button("🎬 Load Custom Clip", type="primary"):
                    st.session_state.current_clip_time = (start_time, end_time)
                    st.session_state.clip_chat_history = []
                    st.rerun()
            
            # =================================================================
            # SHOW CLIP & QA
            # =================================================================
            if st.session_state.current_clip_time:
                clip_start, clip_end = st.session_state.current_clip_time
                
                st.markdown("---")
                st.markdown(f"## 🎥 Current Clip: {clip_start:.1f}s - {clip_end:.1f}s ({clip_end - clip_start:.1f}s)")
                
                # Show video clip
                if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
                    st.markdown("### 📹 Video Clip")
                    
                    clips_dir = "extracted_clips"
                    os.makedirs(clips_dir, exist_ok=True)
                    
                    clip_filename = f"clip_{clip_start:.0f}_{clip_end:.0f}.mp4"
                    clip_path = os.path.join(clips_dir, clip_filename)
                    
                    with st.spinner("Extracting clip..."):
                        success = extract_video_clip(
                            st.session_state.video_file_path,
                            clip_start,
                            clip_end,
                            clip_path
                        )
                    
                    if success:
                        st.video(clip_path)
                        
                        with open(clip_path, 'rb') as f:
                            st.download_button(
                                "📥 Download This Clip",
                                f,
                                file_name=clip_filename,
                                mime="video/mp4"
                            )
                    else:
                        st.warning("Could not extract video clip, but you can still ask questions!")
                else:
                    st.warning("⚠️ Video file not available. You can still ask questions based on frame descriptions!")
                
                # Show frame descriptions
                st.markdown("---")
                st.markdown("### 📝 Frame Descriptions in This Clip")
                
                clip_frames = [f for f in frames if clip_start <= f['timestamp_seconds'] <= clip_end]
                
                if clip_frames:
                    for frame in clip_frames:
                        with st.expander(f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s)"):
                            st.markdown(frame['description'])
                else:
                    st.warning("No analyzed frames in this time window.")
                
                st.markdown("---")
                
                # =============================================================
                # Q&A INTERFACE
                # =============================================================
                st.markdown("### 💬 Ask Questions About This Clip")
                
                with st.expander("💡 Example Questions"):
                    st.markdown("""
                    **About Content:**
                    - "How is this problem solved?"
                    - "What equation is shown?"
                    - "What's the trick mentioned?"
                    
                    **Generate Content:**
                    - "Generate 4 similar questions"
                    - "Create practice problems like this"
                    - "Make flashcards from this"
                    
                    **Explain:**
                    - "Show me step-by-step solution"
                    - "Create a flowchart"
                    - "Break this down simply"
                    """)
                
                # Display chat history
                for chat in st.session_state.clip_chat_history:
                    with st.chat_message("user"):
                        st.write(chat["question"])
                    with st.chat_message("assistant"):
                        st.markdown(chat["answer"])
                
                # Chat input
                user_question = st.chat_input(f"Ask about the {clip_start:.1f}s-{clip_end:.1f}s clip...")
                
                if user_question:
                    with st.chat_message("user"):
                        st.write(user_question)
                    
                    history_str = ""
                    for chat in st.session_state.clip_chat_history:
                        history_str += f"USER: {chat['question']}\nASSISTANT: {chat['answer']}\n\n"
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Analyzing clip..."):
                            answer = chat_with_video_clip(
                                user_question,
                                st.session_state.video_data,
                                st.session_state.text_model,
                                st.session_state.text_tokenizer,
                                clip_start,
                                clip_end,
                                history_str
                            )
                        st.markdown(answer)
                    
                    st.session_state.clip_chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
                
                # Action buttons
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.clip_chat_history = []
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Select Different Clip"):
                        st.session_state.current_clip_time = None
                        st.session_state.clip_chat_history = []
                        st.rerun()
            
            else:
                st.info("👆 Select a clip above to start asking questions!")


if __name__ == "__main__":
    main()