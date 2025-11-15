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
    """Extract frames from video at specified interval, with temporal context frames"""
    frames_data = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # First pass: extract all frames into a buffer
    all_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_count += 1

    cap.release()

    # Second pass: create temporal windows (before, current, after)
    second_count = 0
    for i in range(0, len(all_frames), frame_interval):
        second_count += 1

        # Get temporal context frames (frame before, current, frame after)
        temporal_frames = []

        # Frame N-1 (before) - 15 frames before current
        before_idx = max(0, i - 15)
        frame_rgb = cv2.cvtColor(all_frames[before_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        # Frame N (current)
        frame_rgb = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        # Frame N+1 (after) - 15 frames after current
        after_idx = min(len(all_frames) - 1, i + 15)
        frame_rgb = cv2.cvtColor(all_frames[after_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        frames_data.append({
            "frame_number": second_count,
            "images": temporal_frames  # Now contains 3 images instead of 1
        })

    return frames_data, fps, total_frames

def process_frame_batch(frames_batch, model, processor, previous_descriptions=None):
    """Process a batch of frames with the model, using temporal context"""
    if not frames_batch:
        return []

    results = []

    # Process frames in batch
    for frame_data in frames_batch:
        frame_num = frame_data["frame_number"]
        temporal_images = frame_data["images"]  # List of 3 images: [before, current, after]

        # Multi-frame temporal prompt
        prompt = "You are viewing 3 consecutive frames from a video (before, current, after). Describe what ACTION or MOVEMENT is occurring in the middle frame. Focus on: 1) What the main subjects are DOING (not just their appearance), 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."

        # Build conversation with all 3 images
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame BEFORE:"},
                    {"type": "image", "image": temporal_images[0]},
                    {"type": "text", "text": "Frame CURRENT (describe this one):"},
                    {"type": "image", "image": temporal_images[1]},
                    {"type": "text", "text": "Frame AFTER:"},
                    {"type": "image", "image": temporal_images[2]},
                    {"type": "text", "text": prompt},
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

        # Decode only the generated tokens (not the entire prompt)
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Clean up any residual formatting
        description = description.strip()

        results.append({
            "frame_number": frame_num,
            "text": description
        })

    return results

def save_incremental_json(video_name, all_results, output_dir, is_final=False):
    """Save results to JSON file incrementally"""
    output_data = {
        "video_name": video_name,
        "total_frames": len(all_results),
        "frames": sorted(all_results, key=lambda x: x["frame_number"]),
        "status": "complete" if is_final else "in_progress"
    }

    json_filename = os.path.join(output_dir, "video_analysis.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    return json_filename

def process_video(video_path, model, processor, video_name, output_dir, batch_size=4, max_workers=2, save_interval=8):
    """Process video and extract frame descriptions with parallel processing and incremental saving"""

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
    last_save_count = 0

    # Step 3: Process batches sequentially
    for batch_idx, batch in enumerate(batches):
        try:
            batch_results = process_frame_batch(batch, model, processor)
            all_results.extend(batch_results)
            processed_count += len(batch_results)

            # Update progress
            progress = processed_count / num_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing: {processed_count}/{num_frames} frames complete")

            # Save incrementally every N frames
            if processed_count - last_save_count >= save_interval:
                json_filename = save_incremental_json(video_name, all_results, output_dir, is_final=False)
                last_save_count = processed_count
                status_text.text(f"Processing: {processed_count}/{num_frames} frames complete | 💾 Saved checkpoint")

        except Exception as e:
            st.error(f"Error processing batch {batch_idx}: {str(e)}")

    # Final save with complete status
    json_filename = save_incremental_json(video_name, all_results, output_dir, is_final=True)

    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Analyzed {len(all_results)} frames")

    return all_results, json_filename

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
            # Create output directory
            output_dir = "video_frames_analysis"
            os.makedirs(output_dir, exist_ok=True)

            # Load model
            with st.spinner("Loading model..."):
                model, processor = load_model()
            st.success("✓ Model loaded!")

            # Process video with performance settings and incremental saving
            result = process_video(
                video_path,
                model,
                processor,
                video_name=uploaded_file.name,
                output_dir=output_dir,
                batch_size=batch_size,
                max_workers=max_workers,
                save_interval=8
            )

            if result:
                results, json_filename = result
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

                # Prepare data for download button
                output_data = {
                    "video_name": uploaded_file.name,
                    "total_frames": len(results),
                    "frames": results,
                    "status": "complete"
                }
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