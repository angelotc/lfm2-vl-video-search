import cv2
import json
import os
import streamlit as st
import tempfile
from pathlib import Path
import yt_dlp
import hashlib

# Import from embeddings package
from embeddings import (
    load_model,
    load_embedding_model,
    process_video,
    search_frames,
    get_frame_timestamp,
    extract_clip
)


def hash_youtube_url(url):
    """
    Create a hash of the YouTube URL to use as a unique identifier.

    Args:
        url: YouTube video URL

    Returns:
        str: SHA256 hash of the URL (first 16 characters)
    """
    return hashlib.sha256(url.encode()).hexdigest()[:16]


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


def main():
    st.set_page_config(page_title="Semantic Video Search", page_icon="🎬", layout="wide")

    st.title("Semantic Video Search")

    # Initialize session state for YouTube video
    if 'youtube_video_path' not in st.session_state:
        st.session_state.youtube_video_path = None
    if 'youtube_video_name' not in st.session_state:
        st.session_state.youtube_video_name = None
    if 'youtube_url' not in st.session_state:
        st.session_state.youtube_url = ""
    if 'youtube_url_hash' not in st.session_state:
        st.session_state.youtube_url_hash = None

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload File", "🔗 YouTube URL"],
        horizontal=True
    )

    uploaded_file = None
    video_path_from_youtube = None
    video_name_from_youtube = None

    if input_method == "📁 Upload File":
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])
        # Clear YouTube video if switching to file upload
        if st.session_state.youtube_video_path is not None:
            st.session_state.youtube_video_path = None
            st.session_state.youtube_video_name = None
    else:
        # YouTube URL input - use session state to persist the URL
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            value=st.session_state.youtube_url,
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input"
        )

        # Update session state when URL changes
        if youtube_url != st.session_state.youtube_url:
            st.session_state.youtube_url = youtube_url

        if youtube_url:
            # Calculate hash for this URL
            url_hash = hash_youtube_url(youtube_url)

            # Check if embeddings already exist for this URL
            output_dir = "video_frames_analysis"
            os.makedirs(output_dir, exist_ok=True)
            hash_json_filename = os.path.join(output_dir, f"yt_{url_hash}.json")

            embeddings_exist = os.path.exists(hash_json_filename)

            if embeddings_exist:
                st.success(f"🎯 Found existing analysis for this YouTube video!")
                st.info("💡 You can load the existing analysis without re-downloading the video.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Load Existing Analysis", type="primary"):
                        # Load JSON to get video name
                        with open(hash_json_filename, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)

                        # Set session state as if video was downloaded
                        st.session_state.youtube_video_name = existing_data.get('video_name', f'YouTube Video {url_hash}')
                        st.session_state.youtube_url_hash = url_hash
                        # Note: video_path will be None, but we don't need it for search
                        st.success("✓ Loaded existing analysis!")
                        st.rerun()

                with col2:
                    download_anyway = st.button("🔽 Download Video Anyway")
            else:
                download_anyway = st.button("🔽 Download Video", type="primary")

            if embeddings_exist == False and download_anyway or (embeddings_exist and download_anyway):
                with st.spinner("⏳ Downloading from YouTube..."):
                    downloaded_path, downloaded_name = download_youtube_video(youtube_url)

                if downloaded_path:
                    # Store in session state including hash
                    st.session_state.youtube_video_path = downloaded_path
                    st.session_state.youtube_video_name = downloaded_name
                    st.session_state.youtube_url_hash = url_hash
                    st.success(f"✓ Downloaded: {downloaded_name}")
                    st.rerun()  # Rerun to update the UI

        # Use session state values - ALWAYS display if available
        if st.session_state.youtube_video_path or st.session_state.youtube_url_hash:
            video_path_from_youtube = st.session_state.youtube_video_path
            video_name_from_youtube = st.session_state.youtube_video_name

            if video_path_from_youtube:
                st.info(f"📹 Video loaded: {video_name_from_youtube}")
                st.video(video_path_from_youtube)
            elif st.session_state.youtube_url_hash:
                # Embeddings loaded without video file
                st.info(f"📊 Using existing embeddings: {video_name_from_youtube}")
                st.info("💡 Video file not downloaded (embeddings only mode)")

    # Determine which video source to use
    # Video is ready if we have an uploaded file, downloaded YouTube video, OR loaded YouTube embeddings
    video_ready = (uploaded_file is not None or
                   video_path_from_youtube is not None or
                   st.session_state.youtube_url_hash is not None)

    if video_ready:
        # Set current video name for both sources
        if uploaded_file is not None:
            current_video_name = uploaded_file.name
        else:
            current_video_name = video_name_from_youtube

        # CPU-only processing (no device selection needed)
        selected_device_type = "cpu"

        # Check if JSON already exists
        output_dir = "video_frames_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # For YouTube videos, use URL hash as filename; for uploads, use video name
        if video_path_from_youtube and st.session_state.youtube_url_hash:
            json_filename = os.path.join(output_dir, f"yt_{st.session_state.youtube_url_hash}.json")
        else:
            base_name = Path(current_video_name).stem
            json_filename = os.path.join(output_dir, f"{base_name}.json")

        json_exists = os.path.exists(json_filename)

        if json_exists:
            # Load existing analysis
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            if video_path_from_youtube:
                st.success(f"✓ Found existing analysis for this YouTube video ({existing_data['total_frames']} frames)")
                st.info(f"📁 Using cached embeddings: yt_{st.session_state.youtube_url_hash}.json")
            else:
                base_name = Path(current_video_name).stem
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
            st.header("Search Video")

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

                    # Check if we have access to video file for clip extraction
                    video_path = None
                    if video_path_from_youtube:
                        video_path = video_path_from_youtube
                    elif uploaded_file:
                        video_path = os.path.join(clips_dir, f"temp_{uploaded_file.name}")
                        if not os.path.exists(video_path):
                            uploaded_file.seek(0)
                            with open(video_path, 'wb') as f:
                                f.write(uploaded_file.read())

                    # Determine if we can extract clips
                    can_extract_clips = video_path is not None and os.path.exists(video_path)

                    if can_extract_clips:
                        # Get video FPS
                        cap = cv2.VideoCapture(video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                    else:
                        st.warning("⚠️ Video file not available - showing search results without video clips")
                        st.info("💡 To extract clips, download the video first")

                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"#### Result {i}: Frame {result['frame_number']}")

                            col1, col2 = st.columns([2, 3])

                            with col1:
                                st.metric("Similarity Score", f"{result['similarity']:.4f}")
                                st.markdown(f"**Description:** {result['text']}")

                            with col2:
                                # Get timestamp from stored data
                                frame_timestamp = get_frame_timestamp(
                                    result['frame_number'],
                                    existing_data
                                )

                                st.markdown(f"**Timestamp:** {frame_timestamp:.2f}s")

                                # Only extract clip if we have video file
                                if can_extract_clips:
                                    # Calculate clip times
                                    start_time = max(0, frame_timestamp - padding_seconds)
                                    end_time = frame_timestamp + padding_seconds

                                    st.markdown(f"**Clip Range:** {start_time:.2f}s - {end_time:.2f}s")

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
                                else:
                                    st.info("📹 Video clip not available (embeddings only mode)")

                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error during search: {e}")

            # Ask if user wants to re-process
            st.markdown("---")
            st.warning("⚠️ Want to re-process this video?")
            reprocess = st.checkbox("Yes, re-process this video")

            if not reprocess:
                st.stop()


        # Performance settings
        with st.expander("⚙️ Performance Settings"):
            st.info("ℹ️ Batch size controls checkpoint frequency. Higher values = fewer saves, but all frames are processed individually to save memory.")

            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider(
                    "Save Checkpoint Every N Frames",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="How often to save progress. Each frame is processed individually."
                )
            with col2:
                max_workers = st.slider(
                    "Worker Threads",
                    min_value=1,
                    max_value=4,
                    value=1,
                    help="Number of parallel workers (currently not used)"
                )

            # Frame fusion option
            st.markdown("---")
            fuse_frames = st.checkbox(
                "⚡ Enable Frame Fusion (~3x speedup)",
                value=False,
                help="Combine 3 temporal frames into a single composite image. Reduces processing time by ~3x but may slightly affect description quality."
            )
            if fuse_frames:
                st.success("✅ Frame fusion enabled: Processing 1 image per frame instead of 3")
            else:
                st.info("ℹ️ Standard mode: Processing 3 images per frame (before, current, after)")

            st.success("✅ CPU-based processing: Each frame processed individually")

        # Start button
        if st.button("🚀 Create Embeddings", type="primary"):
            # Determine video path
            if video_path_from_youtube:
                # Use YouTube video directly
                video_path = video_path_from_youtube
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
            # Create output directory
            output_dir = "video_frames_analysis"
            os.makedirs(output_dir, exist_ok=True)

            # Load models
            with st.spinner("Loading vision-language model on CPU..."):
                model, processor, device_config = load_model(device_type="cpu")
            st.success("✓ VLM loaded on CPU!")

            with st.spinner("Loading embedding model..."):
                embedding_model = load_embedding_model()
            st.success("✓ Embedding model loaded!")

            # Process video with performance settings and incremental saving
            result = process_video(
                video_path,
                model,
                processor,
                embedding_model,
                video_name=current_video_name,
                output_dir=output_dir,
                device_config=device_config,
                batch_size=batch_size,
                max_workers=max_workers,
                save_interval=2,
                fuse_frames=fuse_frames
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
                    "video_name": current_video_name,
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

                # Clean up temp file (but not YouTube downloads)
                if not video_path_from_youtube:
                    os.unlink(video_path)
    

if __name__ == "__main__":
    main()