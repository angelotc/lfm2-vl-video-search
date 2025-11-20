"""
Video processing and embedding generation utilities.

This module handles:
- Frame extraction with temporal context
- Batch processing with VLM
- Embedding generation
- Incremental saving to JSON
"""

import cv2
import json
import os
import time
import torch
import streamlit as st
from PIL import Image
from pathlib import Path


def extract_frames(video_path, sample_interval_seconds=2.0, temporal_offset_seconds=0.8, target_size=(384, 384), fuse_frames=False):
    """Extract frames from video at specified time interval, with temporal context frames

    Uses frame seeking to avoid loading entire video into memory - much more memory efficient!

    Args:
        video_path: Path to the video file
        sample_interval_seconds: Time interval between sampled frames (default: 2.0 seconds)
        temporal_offset_seconds: Time offset for before/after frames (default: 0.8 seconds)
        target_size: Tuple (width, height) to resize frames for faster processing (default: (384, 384))
        fuse_frames: If True, fuse 3 temporal frames into single composite image for ~3x speedup (default: False)
    """
    frames_data = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame intervals based on time
    sample_frame_interval = int(sample_interval_seconds * fps)  # Frames between samples
    temporal_frame_offset = int(temporal_offset_seconds * fps)  # Frames offset for before/after

    # Helper function to read a specific frame by seeking
    def read_frame_at_index(frame_idx):
        """Read a single frame at specific index using seeking"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            if target_size:
                frame_pil = frame_pil.resize(target_size, Image.LANCZOS)
            return frame_pil
        return None

    # Stream frames: only load what we need when we need it
    sample_count = 0
    for i in range(0, total_frames, sample_frame_interval):
        sample_count += 1

        # Get temporal context frames (frame before, current, after)
        temporal_frames = []

        # Frame N-1 (before) - temporal_offset_seconds before current
        before_idx = max(0, i - temporal_frame_offset)
        frame_before = read_frame_at_index(before_idx)
        if frame_before is None:
            continue
        temporal_frames.append(frame_before)

        # Frame N (current)
        frame_current = read_frame_at_index(i)
        if frame_current is None:
            continue
        temporal_frames.append(frame_current)

        # Frame N+1 (after) - temporal_offset_seconds after current
        after_idx = min(total_frames - 1, i + temporal_frame_offset)
        frame_after = read_frame_at_index(after_idx)
        if frame_after is None:
            continue
        temporal_frames.append(frame_after)

        # Only add if we successfully got all 3 frames
        if len(temporal_frames) == 3:
            # Optionally fuse the 3 frames into a single composite image
            if fuse_frames:
                # Create side-by-side composite: [before|current|after]
                width, height = temporal_frames[0].size
                composite = Image.new('RGB', (width * 3, height))
                composite.paste(temporal_frames[0], (0, 0))
                composite.paste(temporal_frames[1], (width, 0))
                composite.paste(temporal_frames[2], (width * 2, 0))
                images_to_store = [composite]  # Single fused image
            else:
                images_to_store = temporal_frames  # 3 separate images

            frames_data.append({
                "frame_number": sample_count,
                "actual_frame_index": i,  # Store the actual frame index in the video
                "timestamp": i / fps,  # Store the timestamp in seconds
                "images": images_to_store  # Either 1 fused image or 3 separate images
            })

    cap.release()
    return frames_data, fps, total_frames


def process_frame_batch(frames_batch, model, processor, embedding_model, previous_descriptions=None):
    """Process a batch of frames with the model, using temporal context and true batch inference"""
    if not frames_batch:
        return []

    # Multi-frame temporal prompt (shared across all frames)
    prompt = "You are viewing 3 consecutive frames from a video (before, current, after). Describe what ACTION or MOVEMENT is occurring in the middle frame. Focus on: 1) What the main subjects are DOING (not just their appearance), 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."

    # MEMORY OPTIMIZATION: Process frames one-by-one instead of true batching
    # This prevents massive memory allocation when batch_size > 1
    results = []

    for frame_data in frames_batch:
        frame_num = frame_data["frame_number"]
        temporal_images = frame_data["images"]  # Either 1 fused image or 3 separate images

        # Build conversation based on whether frames are fused or not
        if len(temporal_images) == 1:
            # Fused mode: single composite image
            fused_prompt = "This image shows 3 consecutive video frames arranged left-to-right (BEFORE | CURRENT | AFTER). Describe what ACTION or MOVEMENT is occurring in the MIDDLE frame. Focus on: 1) What the main subjects are DOING, 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temporal_images[0]},
                        {"type": "text", "text": fused_prompt},
                    ],
                },
            ]
        else:
            # Unfused mode: 3 separate images
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

        # Process ONE frame at a time to avoid memory explosion
        # Use inference_mode and explicit cleanup for memory efficiency
        inputs = None
        outputs = None

        try:
            with torch.inference_mode():
                # Apply chat template for single frame
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )

                # Move inputs to the same device as model
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}

                # Generate for this single frame
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.08,
                    top_p=0.95,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None,
                )

                # Decode output immediately while still in context
                prompt_length = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][prompt_length:]
                description = processor.decode(generated_ids, skip_special_tokens=True)

            # Add result for this frame
            results.append({
                "frame_number": frame_data["frame_number"],
                "timestamp": frame_data["timestamp"],
                "actual_frame_index": frame_data["actual_frame_index"],
                "text": description.strip()
            })

        finally:
            # CRITICAL: Explicitly delete tensors to free memory
            # Delete inputs
            if inputs is not None:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        del v
                del inputs

            # Delete outputs
            if outputs is not None:
                del outputs

            # Delete conversation to free image references
            del conversation

    # Generate embeddings for all descriptions in this batch
    descriptions_text = [r["text"] for r in results]
    embeddings = embedding_model.encode(descriptions_text, convert_to_numpy=True)

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


def process_video(video_path, model, processor, embedding_model, video_name, output_dir, device_config, batch_size=4, max_workers=2, save_interval=8, fuse_frames=False):
    """Process video and extract frame descriptions with parallel processing and incremental saving

    Args:
        fuse_frames: If True, fuse 3 temporal frames into 1 composite image for ~3x speedup (default: False)
    """

    # Step 1: Extract all frames first
    status_text = st.empty()
    status_text.text("📹 Extracting frames from video...")

    # Sample every 2 seconds with ±0.8 second temporal context
    frames_data, fps, total_frames = extract_frames(
        video_path,
        sample_interval_seconds=1.5,
        temporal_offset_seconds=0.8,
        fuse_frames=fuse_frames
    )

    if frames_data is None:
        st.error("Error: Could not open video file")
        return None

    num_frames = len(frames_data)
    st.info(f"📹 FPS: {fps:.2f}, extracted {num_frames} frames for processing")

    # Step 2: Split frames into batches
    batches = [frames_data[i:i + batch_size] for i in range(0, num_frames, batch_size)]

    # Progress bar
    progress_bar = st.progress(0)
    device_name = "GPU" if device_config["device"] != "cpu" else "CPU"
    status_text.text(f"🚀 Processing {num_frames} frames on {device_name} with batch size {batch_size}...")

    all_results = []
    processed_count = 0
    last_save_count = 0

    # Profiling variables
    start_time = time.time()

    # Step 3: Process batches sequentially
    for batch_idx, batch in enumerate(batches):
        try:
            batch_results = process_frame_batch(batch, model, processor, embedding_model)
            all_results.extend(batch_results)
            processed_count += len(batch_results)

            # Calculate profiling metrics
            elapsed_time = time.time() - start_time
            frames_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            seconds_per_frame = elapsed_time / processed_count if processed_count > 0 else 0
            remaining_frames = num_frames - processed_count
            estimated_remaining_time = remaining_frames * seconds_per_frame if processed_count > 0 else 0

            # Format time remaining
            if estimated_remaining_time >= 60:
                time_str = f"{estimated_remaining_time / 60:.1f} min"
            else:
                time_str = f"{estimated_remaining_time:.0f}s"

            # Update progress
            progress = processed_count / num_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(
                f"Processing: {processed_count}/{num_frames} frames | "
                f"⚡ {frames_per_second:.2f} fps ({seconds_per_frame:.1f}s/frame) | "
                f"⏱️ ETA: {time_str}"
            )

            # Save incrementally every N frames
            if processed_count - last_save_count >= save_interval:
                json_filename = save_incremental_json(video_name, all_results, output_dir, is_final=False)
                last_save_count = processed_count
                status_text.text(
                    f"Processing: {processed_count}/{num_frames} frames | "
                    f"⚡ {frames_per_second:.2f} fps ({seconds_per_frame:.1f}s/frame) | "
                    f"⏱️ ETA: {time_str} | 💾 Saved checkpoint"
                )

        except Exception as e:
            st.error(f"Error processing batch {batch_idx}: {str(e)}")

    # Final save with complete status
    json_filename = save_incremental_json(video_name, all_results, output_dir, is_final=True)

    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = len(all_results) / total_time if total_time > 0 else 0
    avg_seconds_per_frame = total_time / len(all_results) if len(all_results) > 0 else 0

    progress_bar.progress(1.0)
    status_text.text(
        f"✅ Complete! Processed {len(all_results)} frames in {total_time:.1f}s | "
        f"⚡ Average: {avg_fps:.2f} fps ({avg_seconds_per_frame:.1f}s/frame)"
    )

    return all_results, json_filename
