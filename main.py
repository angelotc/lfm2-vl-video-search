import cv2
import json
import os
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import tempfile

@st.cache_resource
def load_model(model_id="LiquidAI/LFM2-VL-1.6B"):
    """Load the vision-language model and processor"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="bfloat16"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

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
            status_text.text(f"Processing frame {second_count}...")
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Create conversation for this frame - asking for keywords
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this frame in 10 words or less using keywords only."},
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
            
            # Add to results
            all_frames.append({
                "frame_number": second_count,
                "text": description
            })
        
        frame_count += 1
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Analyzed {second_count} frames")
    
    return all_frames

def main():
    st.set_page_config(page_title="Video Frame Analyzer", page_icon="🎬", layout="wide")
    
    st.title("🎬 Video Frame Analyzer")
    st.markdown("Upload a video to generate keywords for each frame")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])
    
    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)
        
        # Start button
        if st.button("🚀 Start Analysis", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            -m ""
            # Load model
            with st.spinner("Loading model..."):
                model, processor = load_model()
            st.success("✓ Model loaded!")
            
            # Process video
            results = process_video(video_path, model, processor)
            
            if results:
                # Create output directory
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save to single JSON file
                output_data = {
                    "video_name": uploaded_file.name,
                    "total_frames": len(results),
                    "frames": results
                }
                
                json_filename = os.path.join(output_dir, "video_analysis.json")
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                
                st.success(f"✓ Saved to: {json_filename}")
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Show as table
                for frame in results:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**Frame {frame['frame_number']}**")
                    with col2:
                        st.markdown(frame['text'])
                
                # Download button
                json_str = json.dumps(output_data, indent=4)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="video_analysis.json",
                    mime="application/json"
                )
                
                # Clean up temp file
                os.unlink(video_path)
    
    else:
        st.info("👆 Upload a video file to get started")

if __name__ == "__main__":
    main()