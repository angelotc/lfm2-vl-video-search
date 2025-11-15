import cv2
import json
import os
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import tempfile
import yt_dlp

@st.cache_resource
def load_vision_model(model_id="LiquidAI/LFM2-VL-450M"):
    """Load the vision-language model and processor"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="bfloat16"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

@st.cache_resource
def load_text_model(model_id="LiquidAI/LFM2-350M"):
    """Load the text model for chat"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def download_youtube_video(url, output_path=None):
    """Download video from YouTube URL"""
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

def process_video(video_path, model, processor, smart_filtering=True):
    """Process video and extract frame descriptions with intelligent VLM-based filtering"""
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
    
    st.info(f"📹 FPS: {fps:.2f}, extracting 1 frame per second")
    if smart_filtering:
        st.info(f"🧠 Smart filtering enabled - VLM checks for changes")
    
    frame_count = 0
    second_count = 0
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Process every Nth frame (where N = fps, so 1 frame per second)
        if frame_count % frame_interval == 0:
            second_count += 1
            timestamp = frame_count / fps
            
            status_text.text(f"Processing frame {second_count} at {timestamp:.2f}s...")
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # For first 5 frames, always add them
            if len(all_frames) < 5:
                # Standard description prompt
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe what you see in this frame. Focus on: 1) Any text, numbers, equations, or mathematical expressions. 2) Any solutions, answers, or calculations. 3) Key objects or content. Be specific."},
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
                
                # Get input length to skip it later
                input_length = inputs['input_ids'].shape[1]
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.5
                )
                
                # Decode only the NEW tokens (skip input)
                new_tokens = outputs[0][input_length:]
                description = processor.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Add to results
                all_frames.append({
                    "frame_number": second_count,
                    "timestamp_seconds": round(timestamp, 2),
                    "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                    "description": description
                })
                
                status_text.text(f"✓ Frame {second_count} added (building baseline)")
                
            else:
                # From frame 6 onwards: Check if different from previous 5
                if smart_filtering:
                    # Get last 5 frames context - use shorter snippets
                    previous_context = ""
                    for prev_frame in all_frames[-5:]:
                        previous_context += f"Frame {prev_frame['frame_number']}: {prev_frame['description'][:80]}...\n"
                    
                    # Ask VLM to compare - improved prompt
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": f"""Previous frames summary:
{previous_context}

Look at the current image. Does it show NEW or DIFFERENT content?
- New text, numbers, equations, or calculations?
- Different step in solving a problem?
- New visual elements or changes?

Answer 'YES' if there's ANY new or different content. Answer 'NO' only if it's almost identical."""},
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
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.5
                    )
                    
                    # Decode only the NEW tokens
                    new_tokens = outputs[0][input_length:]
                    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    # Check if VLM says it's different - more lenient
                    response_upper = response.upper()
                    is_different = "YES" in response_upper[:100] or "NEW" in response_upper[:100] or "DIFFERENT" in response_upper[:100]
                    
                    if is_different:
                        # Frame is different - get full description
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": "Describe what you see in this frame. Focus on: 1) Any text, numbers, equations, or mathematical expressions. 2) Any solutions, answers, or calculations. 3) Key objects or content. Be specific."},
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
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=0.5
                        )
                        
                        # Decode only the NEW tokens
                        new_tokens = outputs[0][input_length:]
                        description = processor.decode(new_tokens, skip_special_tokens=True).strip()
                        
                        # Add to results
                        all_frames.append({
                            "frame_number": second_count,
                            "timestamp_seconds": round(timestamp, 2),
                            "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                            "description": description
                        })
                        
                        status_text.text(f"✓ Frame {second_count} added (NEW content detected)")
                    else:
                        # Frame is similar - skip it
                        skipped_count += 1
                        status_text.text(f"⏭️ Frame {second_count} skipped (similar to previous) - {skipped_count} skipped")
                
                else:
                    # Smart filtering disabled - add all frames
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "Describe what you see in this frame. Focus on: 1) Any text, numbers, equations. 2) Any solutions or calculations. 3) Key content."},
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
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.5
                    )
                    
                    # Decode only the NEW tokens
                    new_tokens = outputs[0][input_length:]
                    description = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    all_frames.append({
                        "frame_number": second_count,
                        "timestamp_seconds": round(timestamp, 2),
                        "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                        "description": description
                    })
        
        frame_count += 1
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Saved {len(all_frames)} frames, skipped {skipped_count} similar frames")
    
    return all_frames


def chat_with_video(user_question, video_data, text_model, tokenizer):
    """Simple chat - search through video frames using text model"""
    
    # Build context from all frames (concise version)
    context = "VIDEO FRAMES:\n\n"
    
    for frame in video_data['frames']:
        context += f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s):\n"
        context += f"{frame['description']}\n\n"
    
    # Simple prompt
    prompt = f"""{context}

USER QUESTION: {user_question}

ANSWER (provide frame numbers and timestamps if relevant):"""
    
    # Generate answer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(text_model.device)
    
    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the new tokens (skip input)
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    return response

def main():
    st.set_page_config(page_title="Video Frame Analyzer & Chat", page_icon="🎬", layout="wide")
    
    st.title("🎬 Video Frame Analyzer & Chat")
    st.markdown("Upload a video or provide a YouTube URL to analyze frames and chat about the content")
    
    # Initialize session state
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create tabs
    tab1, tab2 = st.tabs(["📹 Video Analysis", "💬 Chat with Video"])
    
    # Tab 1: Video Upload and Processing
    with tab1:
        # Choose input method
        input_method = st.radio("Choose input method:", ["Upload File", "YouTube URL"])
        
        # Smart filtering toggle
        smart_filtering = st.checkbox("🧠 Enable Smart Filtering (VLM checks for frame changes)", value=True)
        
        video_path = None
        video_name = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])
            
            if uploaded_file is not None:
                st.video(uploaded_file)
                video_name = uploaded_file.name
                
                if st.button("🚀 Start Analysis", type="primary", key="upload_analyze"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
        
        else:  # YouTube URL
            youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
            
            if youtube_url:
                if st.button("🚀 Download & Analyze", type="primary", key="youtube_analyze"):
                    with st.spinner("Downloading video from YouTube..."):
                        video_path, video_name = download_youtube_video(youtube_url)
                    
                    if video_path:
                        st.success(f"✓ Downloaded: {video_name}")
                        st.video(video_path)
        
        # Process video if we have a path
        if video_path:
            # Load vision model
            with st.spinner("Loading vision model..."):
                vision_model, processor = load_vision_model()
            st.success("✓ Vision model loaded!")
            
            # Process video (VLM does all the work)
            results = process_video(video_path, vision_model, processor, smart_filtering)
            
            if results:
                # Create output directory
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save to single JSON file
                output_data = {
                    "video_name": video_name,
                    "total_frames": len(results),
                    "smart_filtering_enabled": smart_filtering,
                    "frames": results
                }
                
                # Store in session state
                st.session_state.video_data = output_data
                
                json_filename = os.path.join(output_dir, "video_analysis.json")
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                
                st.success(f"✓ Saved to: {json_filename}")
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Frames Saved", len(results))
                with col2:
                    if smart_filtering:
                        st.metric("Quality", "Filtered for changes")
                
                # Show as table
                for frame in results:
                    col1, col2, col3 = st.columns([1, 1, 5])
                    with col1:
                        st.markdown(f"**Frame {frame['frame_number']}**")
                    with col2:
                        st.markdown(f"`{frame['timestamp_formatted']}`")
                    with col3:
                        st.markdown(frame['description'][:200] + "..." if len(frame['description']) > 200 else frame['description'])
                
                # Download button
                json_str = json.dumps(output_data, indent=4)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="video_analysis.json",
                    mime="application/json"
                )
                
                # Clean up temp file
                if os.path.exists(video_path):
                    os.unlink(video_path)
                
                st.info("✨ Now go to the 'Chat with Video' tab to ask questions!")
        
        if not video_path and input_method == "Upload File":
            st.info("👆 Upload a video file to get started")
        elif not youtube_url and input_method == "YouTube URL":
            st.info("👆 Enter a YouTube URL to get started")
    
    # Tab 2: Chat Interface
    with tab2:
        # Option to load existing JSON
        st.markdown("### 📂 Load Video Analysis")
        uploaded_json = st.file_uploader("Upload existing JSON analysis (optional)", type=['json'], key="json_uploader")
        if uploaded_json:
            loaded_data = json.load(uploaded_json)
            st.session_state.video_data = loaded_data
            st.success(f"✓ Loaded: {loaded_data['video_name']} ({loaded_data['total_frames']} frames)")
        
        if st.session_state.video_data is None:
            st.warning("⚠️ Please analyze a video first or upload a JSON file")
        else:
            st.success(f"✓ Loaded: {st.session_state.video_data['video_name']}")
            st.info(f"📊 {st.session_state.video_data['total_frames']} frames available")
            
            # Load text model
            if 'text_model' not in st.session_state:
                with st.spinner("Loading chat model..."):
                    text_model, text_tokenizer = load_text_model()
                    st.session_state.text_model = text_model
                    st.session_state.text_tokenizer = text_tokenizer
                st.success("✓ Chat model loaded!")
            
            # Chat interface
            st.markdown("---")
            st.subheader("💬 Ask Questions")
            
            st.info("💡 Ask questions about the video - AI will search through all frames")
            
            # Example questions
            with st.expander("📋 Example questions"):
                st.markdown("""
                - "What frame shows the final answer?"
                - "Where is the solution explained?"
                - "What does frame 12 talk about?"
                - "Show me frames with equations"
                - "When does the calculation happen?"
                """)
            
            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
            
            # Chat input
            user_question = st.chat_input("Ask a question about your video...")
            
            if user_question:
                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching..."):
                        answer = chat_with_video(
                            user_question,
                            st.session_state.video_data,
                            st.session_state.text_model,
                            st.session_state.text_tokenizer
                        )
                    st.markdown(answer)
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer
                })
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()