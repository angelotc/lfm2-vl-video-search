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
    extract_clip,
    detect_device
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
        output_path: Directory to save video (saves to youtube_downloads if None)

    Returns:
        tuple: (video_path, video_title) or (None, None) on error
    """
    if output_path is None:
        output_path = "videos"
        os.makedirs(output_path, exist_ok=True)

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
    st.set_page_config(page_title="Semantic Video Search", layout="wide")

    st.title("Semantic Video Search")

    # Create necessary directories
    os.makedirs("videos", exist_ok=True)
    os.makedirs("video_frames_analysis", exist_ok=True)
    os.makedirs("extracted_clips", exist_ok=True)

    # Initialize session state for YouTube video
    if 'youtube_video_path' not in st.session_state:
        st.session_state.youtube_video_path = None
    if 'youtube_video_name' not in st.session_state:
        st.session_state.youtube_video_name = None
    if 'youtube_url' not in st.session_state:
        st.session_state.youtube_url = ""
    if 'youtube_url_hash' not in st.session_state:
        st.session_state.youtube_url_hash = None

    # Initialize session state for E2B sandbox (cloud mode)
    if 'e2b_sandbox' not in st.session_state:
        st.session_state.e2b_sandbox = None

    # Processing Mode Selection
    st.markdown("---")
    st.subheader("Processing Mode")

    processing_mode = st.radio(
        "Choose where to process:",
        options=["Local", "Cloud (E2B + Groq + MongoDB)"],
        index=0,
        horizontal=True,
        help="Local: Process on your computer (CPU/GPU). Cloud: Process using E2B sandboxes with Groq VLM (faster, requires API keys)"
    )

    is_cloud_mode = processing_mode == "Cloud (E2B + Groq + MongoDB)"

    # Device selection - only show for LOCAL mode
    selected_device_type = "cpu"  # Default
    if not is_cloud_mode:
        # Auto-detect GPU/CPU
        detected_device_type, detected_device_name = detect_device()

        # Build device options
        device_options = ["CPU"]
        device_mapping = {"CPU": "cpu"}

        if detected_device_type == "cuda":
            gpu_option = f"GPU ({detected_device_name.split('(')[1].split(')')[0]})"
            device_options.insert(0, gpu_option)  # Add GPU as first option
            device_mapping[gpu_option] = "cuda"
            default_index = 0  # Default to GPU if available
        else:
            default_index = 0  # Only CPU available

        # Radio button for device selection
        selected_device_label = st.radio(
            "Choose processing device:",
            options=device_options,
            index=default_index,
            horizontal=True,
            help="GPU provides faster processing if available. CPU is slower but works on all systems."
        )

        selected_device_type = device_mapping[selected_device_label]

        # Show selected device info
        if detected_device_type == "cuda":
            if selected_device_type == "cuda":
                st.success(f"✓ GPU acceleration enabled: {detected_device_name}")
            else:
                st.info(f"💻 CPU processing (GPU available but not selected)")
        else:
            st.warning("⚠ No GPU detected - using CPU")

    # Check for required environment variables in cloud mode
    if is_cloud_mode:
        import dotenv
        dotenv.load_dotenv()

        missing_vars = []
        e2b_key = os.getenv("E2B_API_KEY", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        mongodb_uri = os.getenv("MONGODB_URI", "").strip()

        if not e2b_key or e2b_key == "your_e2b_api_key_here":
            missing_vars.append("E2B_API_KEY")
        if not groq_key or groq_key == "your_groq_api_key_here":
            missing_vars.append("GROQ_API_KEY")
        if not mongodb_uri or mongodb_uri == "your_mongodb_connection_string_here":
            missing_vars.append("MONGODB_URI")

        if missing_vars:
            st.error(f"❌ Missing or invalid environment variables: {', '.join(missing_vars)}")
            st.info("Please add valid API keys to your .env file. See .env.example for format.")
            st.info("📖 E2B API Key: https://e2b.dev/docs/api-key")
            st.info("📖 Groq API Key: https://console.groq.com/keys")
            st.info("📖 MongoDB URI: https://www.mongodb.com/docs/guides/atlas/connection-string/")
            st.stop()

        st.success("✅ Cloud mode enabled - E2B, Groq, and MongoDB configured")

        # Display sandbox status and cleanup option
        if st.session_state.e2b_sandbox is not None:
            st.info(f"🔄 E2B Sandbox active (reusing for all operations)")
            if st.button("🗑️ Cleanup Sandbox", help="Destroy the current E2B sandbox"):
                try:
                    st.session_state.e2b_sandbox.kill()
                    st.session_state.e2b_sandbox = None
                    st.success("✅ Sandbox cleaned up successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error cleaning up sandbox: {e}")
        else:
            st.info("💡 No active sandbox - will create one when processing video")

    # Input method selection
    st.markdown("---")
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "YouTube URL", "Search From Existing Catalog"],
        horizontal=True
    )

    uploaded_file = None
    video_path_from_youtube = None
    video_name_from_youtube = None
    existing_analysis_loaded = False
    existing_json_path = None

    if input_method == "Upload File":
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mov', 'mp4', 'avi', 'mkv'])
        # Clear YouTube video if switching to file upload
        if st.session_state.youtube_video_path is not None:
            st.session_state.youtube_video_path = None
            st.session_state.youtube_video_name = None
    elif input_method == "YouTube URL":
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
            hash_json_filename = os.path.join(output_dir, f"yt_{url_hash}.json")

            embeddings_exist = os.path.exists(hash_json_filename)

            if embeddings_exist:
                st.success(f"Found existing analysis for this YouTube video!")
                st.info("You can load the existing analysis without re-downloading the video.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load Existing Analysis", type="primary"):
                        # Load JSON to get video name
                        with open(hash_json_filename, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)

                        # Set session state as if video was downloaded
                        st.session_state.youtube_video_name = existing_data.get('video_name', f'YouTube Video {url_hash}')
                        st.session_state.youtube_url_hash = url_hash
                        # Note: video_path will be None, but we don't need it for search
                        st.success("Loaded existing analysis!")
                        st.rerun()

                with col2:
                    download_anyway = st.button("Download Video Anyway")
            else:
                download_anyway = st.button("Download Video", type="primary")

            if embeddings_exist == False and download_anyway or (embeddings_exist and download_anyway):
                with st.spinner("Downloading from YouTube..."):
                    downloaded_path, downloaded_name = download_youtube_video(youtube_url)

                if downloaded_path:
                    # Store in session state including hash
                    st.session_state.youtube_video_path = downloaded_path
                    st.session_state.youtube_video_name = downloaded_name
                    st.session_state.youtube_url_hash = url_hash
                    st.success(f"Downloaded: {downloaded_name}")
                    st.info(f"Video saved to: {downloaded_path}")
                    st.info("✅ All videos are now saved permanently in videos/ directory")
                    st.rerun()  # Rerun to update the UI

        # Use session state values - ALWAYS display if available
        if st.session_state.youtube_video_path or st.session_state.youtube_url_hash:
            video_path_from_youtube = st.session_state.youtube_video_path
            video_name_from_youtube = st.session_state.youtube_video_name

            if video_path_from_youtube:
                st.info(f"Video loaded: {video_name_from_youtube}")
                st.video(video_path_from_youtube)
            elif st.session_state.youtube_url_hash:
                # Embeddings loaded without video file
                st.info(f"Using existing embeddings: {video_name_from_youtube}")
                st.info("Video file not downloaded (embeddings only mode)")

    elif input_method == "Search From Existing Catalog":
        # Show available JSON files
        st.subheader("Select Existing Analysis")

        # Get all JSON files in the analysis directory
        output_dir = "video_frames_analysis"

        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

        if not json_files:
            st.warning("No existing analysis files found. Please process a video first.")
            st.stop()

        # Create a dropdown to select which JSON file to use
        selected_json = st.selectbox(
            "Choose analysis file:",
            json_files,
            help="Select a previously generated video analysis to search through"
        )

        if selected_json:
            existing_json_path = os.path.join(output_dir, selected_json)

            # Load and display info about the selected analysis
            try:
                with open(existing_json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                video_name = existing_data.get('video_name', selected_json)
                total_frames = existing_data.get('total_frames', 0)
                status = existing_data.get('status', 'unknown')

                st.success(f"Loaded: {video_name}")
                st.info(f"Frames: {total_frames} | Status: {status}")

                # Set variables for search interface
                existing_analysis_loaded = True
                current_video_name = video_name

                # Set json_filename for search functionality
                json_filename = existing_json_path

            except Exception as e:
                st.error(f"Error loading analysis file: {e}")
                st.stop()

    # Determine which video source to use
    # Video is ready if we have an uploaded file, downloaded YouTube video, loaded YouTube embeddings, OR existing analysis
    video_ready = (uploaded_file is not None or
                   video_path_from_youtube is not None or
                   st.session_state.youtube_url_hash is not None or
                   existing_analysis_loaded)

    if video_ready:
        # Set current video name for both sources
        if uploaded_file is not None:
            current_video_name = uploaded_file.name
        elif video_path_from_youtube or st.session_state.youtube_url_hash:
            current_video_name = video_name_from_youtube
        # For existing analysis, current_video_name and json_filename are already set above

        # For existing analysis, skip the JSON detection logic
        if not existing_analysis_loaded:
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
        else:
            # For existing analysis, we already have json_filename and the data is loaded
            json_exists = True
            existing_data = json.load(open(existing_json_path, 'r', encoding='utf-8'))

        if json_exists:
            # Load existing analysis (only if not already loaded)
            if not existing_analysis_loaded:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            if video_path_from_youtube:
                st.success(f"Found existing analysis for this YouTube video ({existing_data['total_frames']} frames)")
                st.info(f"Using cached embeddings: yt_{st.session_state.youtube_url_hash}.json")
            elif existing_analysis_loaded:
                st.success(f"Loaded from catalog: {selected_json} ({existing_data['total_frames']} frames)")
            else:
                base_name = Path(current_video_name).stem
                st.success(f"Found existing analysis: {base_name}.json ({existing_data['total_frames']} frames)")

            # Show existing results
            with st.expander("View All Frames", expanded=False):
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
            with st.expander("Clip Settings"):
                padding_seconds = st.slider(
                    "Padding (seconds before/after match)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Seconds to include before and after the matched frame"
                )

            if query:
                # Check if we're in cloud mode with cloud video loaded
                is_cloud_search = 'cloud_video_id' in st.session_state and 'cloud_mongodb_collection' in st.session_state

                if is_cloud_search:
                    # CLOUD SEARCH WITH LLM AGENT
                    from embeddings.cloud import search_video_frames
                    import dotenv
                    dotenv.load_dotenv()

                    collection = st.session_state['cloud_mongodb_collection']
                    video_id = st.session_state['cloud_video_id']
                    groq_api_key = os.getenv("GROQ_API_KEY")

                    with st.spinner("Searching with LLM agent..."):
                        try:
                            results = search_video_frames(
                                query,
                                groq_api_key,
                                collection,
                                video_id,
                                top_k,
                                sandbox=st.session_state.e2b_sandbox  # Pass sandbox for potential future use
                            )
                        except Exception as e:
                            st.error(f"❌ Search error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.stop()
                else:
                    # LOCAL SEARCH WITH EMBEDDINGS
                    # Load embedding model for search (use same device as VLM)
                    with st.spinner("Loading embedding model..."):
                        embedding_model = load_embedding_model(device_type=selected_device_type)

                    # Perform search
                    try:
                        results = search_frames(query, json_filename, embedding_model, top_k)
                    except Exception as e:
                        st.error(f"❌ Search error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()

                st.markdown(f"### Top {len(results)} Results for: '{query}'")

                # Save video temporarily for clip extraction
                clips_dir = "extracted_clips"
                # clips_dir already created in main()

                # Check if we have access to video file for clip extraction
                video_path = None
                videos_dir = "videos"
                # videos_dir already created in main()

                if is_cloud_search:
                    # For cloud mode, use stored video path
                    if 'cloud_video_path' in st.session_state:
                        video_path = st.session_state['cloud_video_path']
                        if not os.path.exists(video_path):
                            video_path = None
                else:
                    # For local mode, find video path
                    if video_path_from_youtube:
                        video_path = video_path_from_youtube
                    elif uploaded_file:
                        # Look for video in videos directory first
                        potential_video_path = os.path.join(videos_dir, uploaded_file.name)
                        if os.path.exists(potential_video_path):
                            video_path = potential_video_path
                        else:
                            # Save to videos directory permanently
                            uploaded_file.seek(0)
                            with open(potential_video_path, 'wb') as f:
                                f.write(uploaded_file.read())
                            video_path = potential_video_path
                    else:
                        # For existing analysis, try to find matching video in videos directory
                        if existing_analysis_loaded and current_video_name:
                            potential_video_path = os.path.join(videos_dir, current_video_name)
                            if os.path.exists(potential_video_path):
                                video_path = potential_video_path
                            else:
                                # Try without extension
                                base_name = Path(current_video_name).stem
                                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                                    potential_video_path = os.path.join(videos_dir, base_name + ext)
                                    if os.path.exists(potential_video_path):
                                        video_path = potential_video_path
                                        break

                # Determine if we can extract clips
                can_extract_clips = video_path is not None and os.path.exists(video_path)

                if can_extract_clips:
                    # Get video FPS
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                else:
                    if existing_analysis_loaded:
                        st.warning("Video file not available in videos/ directory - showing search results without video clips")
                        st.info("Add the matching video file to the videos/ directory to enable clip extraction")
                    else:
                        st.warning("Video file not available - showing search results without video clips")
                        st.info("Upload or download the video to the videos/ directory to enable clip extraction")

                # Display results
                try:
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"#### Result {i}: Frame {result['frame_number']}")

                            col1, col2 = st.columns([2, 3])

                            with col1:
                                # Display score appropriately based on search mode
                                if is_cloud_search:
                                    # Try relevance_score first (from LLM re-ranking), then MongoDB score
                                    score_value = result.get('relevance_score') or result.get('score', 0)
                                    # Convert to float if string
                                    if isinstance(score_value, str):
                                        try:
                                            score_value = float(score_value)
                                        except:
                                            score_value = 0.0
                                    st.metric("Relevance Score", f"{score_value:.4f}")
                                else:
                                    st.metric("Similarity Score", f"{result['similarity']:.4f}")
                                st.markdown(f"**Description:** {result['text']}")

                            with col2:
                                # Get timestamp from stored data
                                if is_cloud_search:
                                    frame_timestamp = result['timestamp']
                                else:
                                    frame_timestamp = get_frame_timestamp(
                                        result['frame_number'],
                                        existing_data
                                    )

                                st.markdown(f"**Timestamp:** {frame_timestamp:.2f}s")

                                # Only show video options if we have video file
                                if can_extract_clips:
                                    # Video display options
                                    video_display_mode = st.radio(
                                        "Video display:",
                                        ["Full Video (at timestamp)", "Extract Clip"],
                                        key=f"display_mode_{i}",
                                        horizontal=True
                                    )

                                    if video_display_mode == "Full Video (at timestamp)":
                                        # Show full video starting at the matched frame timestamp
                                        st.markdown(f"**Playing from:** {frame_timestamp:.2f}s")
                                        st.video(video_path, start_time=int(frame_timestamp))

                                    else:  # Extract Clip
                                        # Adjustable clip time controls
                                        st.markdown("**Adjust Clip Times:**")
                                        clip_col1, clip_col2 = st.columns(2)

                                        with clip_col1:
                                            clip_start = st.number_input(
                                                "Start time (s)",
                                                min_value=0.0,
                                                max_value=frame_timestamp,
                                                value=max(0, frame_timestamp - padding_seconds),
                                                step=0.5,
                                                key=f"clip_start_{i}"
                                            )

                                        with clip_col2:
                                            clip_end = st.number_input(
                                                "End time (s)",
                                                min_value=frame_timestamp,
                                                value=frame_timestamp + padding_seconds,
                                                step=0.5,
                                                key=f"clip_end_{i}"
                                            )

                                        st.markdown(f"**Clip Duration:** {clip_end - clip_start:.2f}s")

                                        # Extract clip button
                                        if st.button(f"Extract Clip", key=f"extract_btn_{i}"):
                                            clip_filename = f"clip_{i}_frame{result['frame_number']}_{query.replace(' ', '_')[:20]}.mp4"
                                            clip_path = os.path.join(clips_dir, clip_filename)

                                            # Show extraction info
                                            with st.spinner(f"Extracting clip {i}..."):
                                                success = extract_clip(video_path, clip_start, clip_end, clip_path)

                                            if success:
                                                # Verify clip exists
                                                if os.path.exists(clip_path):
                                                    file_size = os.path.getsize(clip_path)
                                                    st.success(f"Clip created: {file_size / 1024:.1f} KB")

                                                    # Display video
                                                    st.video(clip_path)

                                                    # Download button
                                                    with open(clip_path, 'rb') as clip_file:
                                                        st.download_button(
                                                            label=f"Download Clip {i}",
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
                                    if existing_analysis_loaded:
                                        st.info("Video clip not available (add video to videos/ directory)")
                                    else:
                                        st.info("Video clip not available (embeddings only mode)")

                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error during search: {e}")

            # Ask if user wants to re-process
            st.markdown("---")
            st.warning("Want to re-process this video?")
            reprocess = st.checkbox("Yes, re-process this video")

            if not reprocess:
                st.stop()


        # Performance settings
        with st.expander("Performance Settings"):
            st.info("Batch size controls checkpoint frequency. Higher values = fewer saves, but all frames are processed individually to save memory.")

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
                "Enable Frame Fusion (~3x speedup)",
                value=False,
                help="Combine 3 temporal frames into a single composite image. Reduces processing time by ~3x but may slightly affect description quality."
            )
            if fuse_frames:
                st.success("Frame fusion enabled: Processing 1 image per frame instead of 3")
            else:
                st.info("Standard mode: Processing 3 images per frame (before, current, after)")

        # Start button
        if st.button("Create Embeddings", type="primary"):
            # Determine video path
            if video_path_from_youtube:
                # Use YouTube video directly
                video_path = video_path_from_youtube
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name

            if is_cloud_mode:
                # CLOUD MODE PROCESSING
                from embeddings.cloud import generate_video_id, setup_mongodb, process_video_cloud
                import dotenv
                dotenv.load_dotenv()

                st.info("☁️ Processing video in cloud using E2B + Groq...")

                # Setup MongoDB
                mongodb_uri = os.getenv("MONGODB_URI")
                mongodb_db = os.getenv("MONGODB_DATABASE", "video_search")
                mongodb_coll = os.getenv("MONGODB_COLLECTION", "video_frames")

                with st.spinner("Connecting to MongoDB..."):
                    mongo_client, collection = setup_mongodb(mongodb_uri, mongodb_db, mongodb_coll)
                st.success("✅ Connected to MongoDB Atlas")

                # Generate video ID
                video_id = generate_video_id(video_path)
                st.info(f"Video ID: {video_id}")

                # Check if video already processed
                existing_count = collection.count_documents({"video_id": video_id})
                if existing_count > 0:
                    st.warning(f"⚠️ Video already processed ({existing_count} frames in database). Re-processing will overwrite existing data.")

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total, msg):
                    progress_bar.progress(current / total if total > 0 else 0)
                    status_text.text(msg)

                # Process video
                try:
                    result = process_video_cloud(
                        video_path=video_path,
                        video_id=video_id,
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        e2b_api_key=os.getenv("E2B_API_KEY"),
                        mongodb_uri=mongodb_uri,
                        mongodb_database=mongodb_db,
                        mongodb_collection=mongodb_coll,
                        progress_callback=update_progress,
                        sandbox=st.session_state.e2b_sandbox  # Reuse existing sandbox
                    )

                    st.success(f"✅ Video processed! {result['total_frames']} frames analyzed.")

                    # Store sandbox in session state for reuse
                    if 'sandbox' in result:
                        st.session_state.e2b_sandbox = result['sandbox']
                        st.info("💾 Sandbox saved for reuse in future operations")

                    # Store video info in session state for search
                    st.session_state['cloud_video_id'] = video_id
                    st.session_state['cloud_video_path'] = video_path
                    st.session_state['cloud_mongodb_collection'] = collection
                    st.session_state['cloud_mongo_client'] = mongo_client

                    # Clean up temp file (but not YouTube downloads)
                    if not video_path_from_youtube:
                        # Don't delete yet - needed for clip extraction
                        pass

                    # Rerun to show search interface
                    st.success("Processing complete! Reloading to show search interface...")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    if 'mongo_client' in locals():
                        mongo_client.close()
                    st.stop()
            else:
                # LOCAL CPU MODE PROCESSING
                # Create output directory
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)

                # Load models with selected device
                device_name_display = "GPU" if selected_device_type == "cuda" else "CPU"
                with st.spinner(f"Loading vision-language model on {device_name_display}..."):
                    model, processor, device_config = load_model(device_type=selected_device_type)
                st.success(f"VLM loaded on {device_name_display}!")

                with st.spinner(f"Loading embedding model on {device_name_display}..."):
                    embedding_model = load_embedding_model(device_type=selected_device_type)
                st.success(f"Embedding model loaded on {device_name_display}!")

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
                    st.success(f"Saved to: {json_filename}")

                    # Display results
                    st.markdown("---")
                    st.header("Results")

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
                        label="Download JSON",
                        data=json_str,
                        file_name="video_analysis.json",
                        mime="application/json"
                    )

                    # Clean up temp file (but not YouTube downloads)
                    if not video_path_from_youtube:
                        os.unlink(video_path)

                    # Rerun to show search interface
                    st.success("Processing complete! Reloading to show search interface...")
                    st.rerun()
    

if __name__ == "__main__":
    main()