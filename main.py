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
from sentence_transformers import SentenceTransformer
from pathlib import Path

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

@st.cache_resource
def load_embedding_model(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence transformer model for embedding text descriptions"""
    embedding_model = SentenceTransformer(model_id)
    return embedding_model

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

def process_frame_batch(frames_batch, model, processor, embedding_model, previous_descriptions=None):
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

    # Generate embeddings for all descriptions in this batch
    descriptions = [r["text"] for r in results]
    embeddings = embedding_model.encode(descriptions, convert_to_numpy=True)

    # Add embeddings to results
    for i, result in enumerate(results):
        result["embedding"] = embeddings[i].tolist()

    return results

def save_incremental_json(video_name, all_results, output_dir, is_final=False):
    """Save results to JSON file incrementally"""
    output_data = {
        "video_name": video_name,
        "total_frames": len(all_results),
        "frames": sorted(all_results, key=lambda x: x["frame_number"]),
        "status": "complete" if is_final else "in_progress"
    }

    # Use video name (without extension) as JSON filename
    base_name = Path(video_name).stem
    json_filename = os.path.join(output_dir, f"{base_name}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    return json_filename

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_frames(query, json_path, embedding_model, top_k=5):
    """Search for frames matching the query"""
    # Load the analysis JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if embeddings exist
    if not data['frames'] or 'embedding' not in data['frames'][0]:
        raise ValueError("No embeddings found in JSON file.")

    # Embed the query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    # Compute similarities
    results = []
    for frame in data['frames']:
        frame_embedding = np.array(frame['embedding'])
        similarity = cosine_similarity(query_embedding, frame_embedding)
        results.append({
            'frame_number': frame['frame_number'],
            'text': frame['text'],
            'similarity': float(similarity)
        })

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def get_frame_timestamp(frame_number, frame_interval=30, fps=30):
    """Calculate timestamp for a frame number"""
    actual_frame_position = (frame_number - 1) * frame_interval
    return actual_frame_position / fps

def extract_clip(video_path, start_time, end_time, output_path):
    """Extract a clip from video between start_time and end_time (in seconds)"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure dimensions are even (required by some codecs)
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        # Try different codecs for better compatibility
        # Use avc1 (H.264) for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            # Fallback to mp4v if avc1 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate frame numbers
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)

        # Set position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_written = 0
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame if dimensions were adjusted
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)
            frames_written += 1
            current_frame += 1

        cap.release()
        out.release()

        if frames_written == 0:
            st.error(f"No frames extracted. Start: {start_time}s, End: {end_time}s")
            return False

        # Verify file was created and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            st.error(f"Clip file not created or empty: {output_path}")
            return False

        return True
    except Exception as e:
        st.error(f"Error extracting clip: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

def process_video(video_path, model, processor, embedding_model, video_name, output_dir, batch_size=4, max_workers=2, save_interval=8):
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
            batch_results = process_frame_batch(batch, model, processor, embedding_model)
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
    st.set_page_config(page_title="Semantic Video Search", page_icon="🎬", layout="wide")

    st.title("🎬 Semantic Video Search")
    st.markdown("Upload a video to generate keywords for each frame")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])

    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)

        # Check if JSON already exists
        output_dir = "video_frames_analysis"
        os.makedirs(output_dir, exist_ok=True)

        base_name = Path(uploaded_file.name).stem
        json_filename = os.path.join(output_dir, f"{base_name}.json")
        json_exists = os.path.exists(json_filename)

        if json_exists:
            # Load existing analysis
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            st.success(f"✓ Found existing analysis: {base_name}.json ({existing_data['total_frames']} frames)")

            # Show existing results
            with st.expander("📊 View All Frames", expanded=False):
                for frame in existing_data['frames']:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**Frame {frame['frame_number']}**")
                    with col2:
                        st.markdown(frame['text'])

            # SEARCH INTERFACE
            st.markdown("---")
            st.header("🔍 Search Video")

            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "Enter your search query",
                    placeholder="e.g., layup, person jumping, basketball shot...",
                    key="search_query"
                )
            with col2:
                top_k = st.number_input("Top results", min_value=1, max_value=20, value=5)

            # Clip settings
            with st.expander("⚙️ Clip Settings"):
                padding_seconds = st.slider(
                    "Padding (seconds before/after match)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Seconds to include before and after the matched frame"
                )

            if query:
                # Load embedding model for search
                with st.spinner("Loading embedding model..."):
                    embedding_model = load_embedding_model()

                # Perform search
                try:
                    results = search_frames(query, json_filename, embedding_model, top_k)

                    st.markdown(f"### 📊 Top {len(results)} Results for: '{query}'")

                    # Save video temporarily for clip extraction
                    clips_dir = "extracted_clips"
                    os.makedirs(clips_dir, exist_ok=True)

                    video_path = os.path.join(clips_dir, f"temp_{uploaded_file.name}")
                    if not os.path.exists(video_path):
                        uploaded_file.seek(0)
                        with open(video_path, 'wb') as f:
                            f.write(uploaded_file.read())

                    # Get video FPS
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"#### Result {i}: Frame {result['frame_number']}")

                            col1, col2 = st.columns([2, 3])

                            with col1:
                                st.metric("Similarity Score", f"{result['similarity']:.4f}")
                                st.markdown(f"**Description:** {result['text']}")

                            with col2:
                                # Calculate timestamp
                                frame_timestamp = get_frame_timestamp(
                                    result['frame_number'],
                                    frame_interval=30,
                                    fps=fps
                                )

                                # Calculate clip times
                                start_time = max(0, frame_timestamp - padding_seconds)
                                end_time = frame_timestamp + padding_seconds

                                st.markdown(f"**Timestamp:** {frame_timestamp:.2f}s (±{padding_seconds}s)")

                                # Extract clip
                                clip_filename = f"clip_{i}_frame{result['frame_number']}_{query.replace(' ', '_')[:20]}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)

                                # Show extraction info
                                with st.spinner(f"Extracting clip {i}..."):
                                    success = extract_clip(video_path, start_time, end_time, clip_path)

                                if success:
                                    # Verify clip exists
                                    if os.path.exists(clip_path):
                                        file_size = os.path.getsize(clip_path)
                                        st.success(f"✓ Clip created: {file_size / 1024:.1f} KB")

                                        # Display video
                                        st.video(clip_path)

                                        # Download button
                                        with open(clip_path, 'rb') as clip_file:
                                            st.download_button(
                                                label=f"📥 Download Clip {i}",
                                                data=clip_file,
                                                file_name=clip_filename,
                                                mime="video/mp4",
                                                key=f"download_{i}"
                                            )
                                    else:
                                        st.error(f"Clip file not found: {clip_path}")
                                else:
                                    st.error("Failed to extract clip (see errors above)")

                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error during search: {e}")

            # Ask if user wants to re-process
            st.markdown("---")
            st.warning("⚠️ Want to re-analyze this video?")
            reprocess = st.checkbox("Yes, re-process this video")

            if not reprocess:
                st.stop()

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

            # Load models
            with st.spinner("Loading vision-language model..."):
                model, processor = load_model()
            st.success("✓ VLM loaded!")

            with st.spinner("Loading embedding model..."):
                embedding_model = load_embedding_model()
            st.success("✓ Embedding model loaded!")

            # Process video with performance settings and incremental saving
            result = process_video(
                video_path,
                model,
                processor,
                embedding_model,
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
    

if __name__ == "__main__":
    main()