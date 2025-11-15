import cv2
import json
import os
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import tempfile
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

@st.cache_resource
def load_model(model_id="LiquidAI/LFM2-VL-450M"):
    """Load the vision-language model and processor"""
    # Determine device
    device = "cpu"

    # For CPU, use float32 for better compatibility
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="bfloat16"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def extract_frames(video_path, frame_interval):
    """Extract frames from video at specified interval"""
    frames_data = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    second_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            second_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames_data.append({
                "frame_number": second_count,
                "image": image
            })

        frame_count += 1

    cap.release()
    return frames_data, fps, total_frames

def process_frame_batch(frames_batch, model, processor):
    """Process a batch of frames with the model"""
    if not frames_batch:
        return []

    results = []

    # Process frames in batch
    for frame_data in frames_batch:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_data["image"]},
                    {"type": "text", "text": "Describe this frame in 10 words or less using keywords only."},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=128)
        description = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        results.append({
            "frame_number": frame_data["frame_number"],
            "text": description
        })

    return results

def process_video(video_path, model, processor, batch_size=4, max_workers=2):
    """Process video and extract frame descriptions with parallel processing"""

    # Step 1: Extract all frames first
    status_text = st.empty()
    status_text.text("📹 Extracting frames from video...")

    frame_interval = 30  # Extract one frame every 30 frames (adjust based on fps)
    frames_data, fps, total_frames = extract_frames(video_path, frame_interval)

    if frames_data is None:
        st.error("Error: Could not open video file")
        return None

    num_frames = len(frames_data)
    st.info(f"📹 FPS: {fps:.2f}, extracted {num_frames} frames for processing")

    # Step 2: Split frames into batches
    batches = [frames_data[i:i + batch_size] for i in range(0, num_frames, batch_size)]

    # Progress bar
    progress_bar = st.progress(0)
    status_text.text(f"🚀 Processing {num_frames} frames with {max_workers} workers...")

    all_results = []
    processed_count = 0

    # Step 3: Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_frame_batch, batch, model, processor): batch_idx
            for batch_idx, batch in enumerate(batches)
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                processed_count += len(batch_results)

                # Update progress
                progress = processed_count / num_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing: {processed_count}/{num_frames} frames complete")

            except Exception as e:
                st.error(f"Error processing batch {batch_idx}: {str(e)}")

    # Sort results by frame number
    all_results.sort(key=lambda x: x["frame_number"])

    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Analyzed {len(all_results)} frames")

    return all_results

def main():
    st.set_page_config(page_title="Video Frame Analyzer", page_icon="🎬", layout="wide")

    st.title("🎬 Video Frame Analyzer")
    st.markdown("Upload a video to generate keywords for each frame")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])

    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)

        # Performance settings
        with st.expander("⚙️ Performance Settings"):
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider(
                    "Batch Size",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="Number of frames to process in each batch. Higher = more memory usage"
                )
            with col2:
                max_workers = st.slider(
                    "Worker Threads",
                    min_value=1,
                    max_value=4,
                    value=2,
                    help="Number of parallel workers. Note: CPU-only models may not benefit from >2 workers"
                )

        # Start button
        if st.button("🚀 Start Analysis", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            # Load model
            with st.spinner("Loading model..."):
                model, processor = load_model()
            st.success("✓ Model loaded!")

            # Process video with performance settings
            results = process_video(video_path, model, processor, batch_size=batch_size, max_workers=max_workers)
            
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