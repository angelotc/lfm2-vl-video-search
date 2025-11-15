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

def process_video(video_path, model, processor):
    """Process video and extract frame descriptions"""
    all_frames = []
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps)  # Extract one frame per second
    
    st.info(f"📹 FPS: {fps:.2f}, extracting 1 frame per second")
    
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
            
            # Updated robust prompt for different types of videos
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "What are the key important elements you see in this frame? Describe any text, objects, people, actions, or educational content present. Use keywords and brief descriptions."},
                    ],
                },
            ]
            
            # Generate description
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=128)
            description = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Add to results with timestamp
            all_frames.append({
                "frame_number": second_count,
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                "description": description
            })
        
        frame_count += 1
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Analyzed {second_count} frames")
    
    return all_frames

def chat_with_video(user_question, video_data, text_model, tokenizer):
    """Chat with the text model using video frame data as context"""
    
    # Prepare simplified context from video frames
    context_summary = f"Video: {video_data['video_name']}\nTotal frames: {video_data['total_frames']}\n\n"
    
    # Build frame information
    frames_text = ""
    for frame in video_data['frames']:
        frames_text += f"[Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s)]: {frame['description']}\n\n"
    
    # Create a focused prompt
    prompt = f"""You are helping a user find specific information in a video by analyzing frame descriptions.

VIDEO INFORMATION:
{context_summary}

FRAME DESCRIPTIONS:
{frames_text}

USER QUESTION: {user_question}

INSTRUCTIONS:
- If the user asks about specific content (like "final answer", "solution", "where it explains X"), find the EXACT frame numbers and timestamps that contain that information
- List the specific frame numbers and timestamps in your response
- Quote the relevant parts from the frame descriptions
- Be concise and direct
- If you can't find the information, say so clearly

Answer:"""
    
    # Generate response
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(text_model.device)
    
    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    
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
            
            # Process video
            results = process_video(video_path, vision_model, processor)
            
            if results:
                # Create output directory
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save to single JSON file
                output_data = {
                    "video_name": video_name,
                    "total_frames": len(results),
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
                
                # Show as table
                for frame in results:
                    col1, col2, col3 = st.columns([1, 1, 5])
                    with col1:
                        st.markdown(f"**Frame {frame['frame_number']}**")
                    with col2:
                        st.markdown(f"`{frame['timestamp_formatted']}`")
                    with col3:
                        st.markdown(frame['description'])
                
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
                
                st.info("✨ Now go to the 'Chat with Video' tab to ask questions about your video!")
        
        if not video_path and input_method == "Upload File":
            st.info("👆 Upload a video file to get started")
        elif not youtube_url and input_method == "YouTube URL":
            st.info("👆 Enter a YouTube URL to get started")
    
    # Tab 2: Chat Interface
    with tab2:
        if st.session_state.video_data is None:
            st.warning("⚠️ Please analyze a video first in the 'Video Analysis' tab")
        else:
            st.success(f"✓ Loaded video: {st.session_state.video_data['video_name']}")
            st.info(f"📊 Total frames: {st.session_state.video_data['total_frames']}")
            
            # Load text model
            if 'text_model' not in st.session_state:
                with st.spinner("Loading chat model..."):
                    text_model, tokenizer = load_text_model()
                    st.session_state.text_model = text_model
                    st.session_state.tokenizer = tokenizer
                st.success("✓ Chat model loaded!")
            
            # Chat interface
            st.markdown("---")
            st.subheader("💬 Ask questions about your video")
            
            # Example questions
            with st.expander("💡 Example questions"):
                st.markdown("""
                - "Pull me the frame where the final answer is shown"
                - "What frames show the final step?"
                - "Which frame explains the solution?"
                - "Show me frames with mathematical equations"
                - "What happens at 00:20?"
                - "Find frames where it explains [specific topic]"
                """)
            
            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
            
            # Chat input
            user_question = st.chat_input("Ask a question about your video...")
            
            if user_question:
                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = chat_with_video(
                            user_question,
                            st.session_state.video_data,
                            st.session_state.text_model,
                            st.session_state.tokenizer
                        )
                    st.write(answer)
                
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