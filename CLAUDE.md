# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based semantic video search system using LiquidAI's LFM2-VL-1.6B vision-language model. The project processes videos frame-by-frame, generates semantic descriptions using the VLM, and enables semantic search capabilities across video content.

**Key capabilities:**
- Upload videos directly or download from YouTube URLs
- Temporal frame analysis with 3-frame context windows
- Action/movement-focused semantic descriptions
- Vector embeddings for semantic search
- Clip extraction with configurable padding
- Incremental processing with checkpoint saves

## Development Commands

### Environment Setup
```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### Running the Applications
```bash
streamlit run app.py   # Basic video upload UI (no AI processing)
streamlit run main.py  # Full video analyzer with search & clip extraction
```

Both apps will open in the browser at `http://localhost:8501`

**Key Features in `main.py`:**
- Auto-detects if video already has JSON analysis (e.g., `wisconsin-vs-montana-clip.json`)
- If analysis exists: shows search interface immediately
- If not: runs VLM analysis first, then shows search
- Search returns clips with ±2s padding and similarity scores

### Searching Video Content

**Option 1: Use the Streamlit UI (Recommended)**
1. Run `streamlit run main.py`
2. Upload video (auto-checks for existing JSON)
3. Enter search query (e.g., "layup", "person jumping")
4. View results with similarity scores and downloadable clips

**Option 2: Use CLI search script**
```bash
python search.py "layup"                           # Find frames matching "layup"
python search.py "person jumping" --top-k 10       # Get top 10 results
python search.py "basketball shot" --json-file path/to/custom_analysis.json
```
Note: CLI shows text results only, no clip extraction

### Required Dependencies
All dependencies are now in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `streamlit`: Web UI framework
- `transformers`, `torch`: HuggingFace model loading
- `sentence-transformers`: Text embedding generation
- `opencv-python` (cv2): Video frame extraction
- `yt-dlp`: YouTube video downloading
- `ffmpeg` (system dependency): Fast clip extraction - must be installed separately and added to PATH

## Architecture

### Project Structure

**Applications:**
- **`main.py`**: Full-featured semantic video search application (Streamlit)
  - YouTube URL download support with URL-based caching
  - Auto-detects existing JSON analysis by video filename or URL hash
  - Two-phase workflow: embedding generation → search
  - Integrated search interface with clip extraction
  - Progress tracking and visualization

- **`app.py`**: Simple video upload and preview UI (no AI processing)
  - Basic file uploader for testing
  - Video preview and metadata display

**Core Modules (`embeddings/` package):**
- **`embeddings/load.py`**: Model loading utilities
  - `load_model()`: Load VLM (LiquidAI/LFM2-VL-1.6B) on CPU with caching
  - `load_embedding_model()`: Load sentence-transformers model with caching

- **`embeddings/embed.py`**: Video processing and embedding generation
  - `extract_frames()`: Temporal frame extraction with configurable intervals
  - `process_frame_batch()`: Frame-by-frame VLM processing with memory optimization
  - `process_video()`: Full video processing pipeline with incremental saves
  - `save_incremental_json()`: Checkpoint saving every N frames

- **`embeddings/query.py`**: Search and clip extraction
  - `search_frames()`: Semantic search using cosine similarity
  - `get_frame_timestamp()`: Convert frame numbers to video timestamps
  - `extract_clip()`: FFmpeg-based clip extraction with fallback re-encoding
  - CLI interface: `python -m embeddings.query "your query"`

**Data Directories (auto-created):**
- **`uploads/`**: Uploaded videos
- **`video_frames_analysis/`**: JSON files with embeddings
  - Uploaded videos: `{video_basename}.json`
  - YouTube videos: `yt_{url_hash}.json` (SHA256 hash of URL)
- **`extracted_clips/`**: Search result video clips
  - Format: `clip_{rank}_frame{number}_{query}.mp4`

### Core Processing Flow (main.py)

1. **Video Input**: User chooses input method
   - **Option A - File Upload**: Upload MP4/AVI/MOV/MKV via file uploader
   - **Option B - YouTube URL**: Enter YouTube URL, downloads with yt-dlp
     - Creates URL hash (SHA256, first 16 chars) for caching
     - Checks for existing analysis: `yt_{url_hash}.json`
     - If analysis exists: can load embeddings without re-downloading video

2. **JSON Detection**: Automatically checks for existing analysis
   - Uploaded videos: looks for `{video_basename}.json`
   - YouTube videos: looks for `yt_{url_hash}.json`
   - If found: Shows search interface immediately (skip to step 7)
   - If not found: Proceeds to analysis (steps 3-6)
3. **Performance Configuration**: User sets batch size and worker threads (if analyzing)
   - Batch size: 1-16 frames per batch (default: 4)
   - Worker threads: 1-4 (displayed for reference, currently uses sequential processing)
4. **Model Loading**: Cached loading via `@st.cache_resource` (if analyzing)
   - **VLM**: `LiquidAI/LFM2-VL-1.6B` from HuggingFace (embeddings/load.py:16)
     - Device: CPU with bfloat16 precision
     - Manual device management (no device_map="auto")
     - Model size: ~3.2GB download
   - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
     - Used to embed text descriptions for semantic search
     - Lightweight model (~90MB) with 384-dimensional embeddings

5. **Temporal Frame Extraction** (Phase 1 - embeddings/embed.py:20-83):
   - Extract ALL frames into memory buffer using OpenCV
   - **Default sampling**: Every 1.5 seconds (configurable via `sample_interval_seconds`)
   - **Temporal context**: ±0.8 seconds around each sample (configurable via `temporal_offset_seconds`)
   - For each sampled frame, create temporal window:
     - Frame N-1: 0.8 seconds before current
     - Frame N: Current frame
     - Frame N+1: 0.8 seconds after current
   - Convert BGR to RGB for model compatibility
   - Store as triplets of PIL Images with timestamp metadata

6. **Frame-by-Frame VLM Processing** (Phase 2 - embeddings/embed.py:86-186):
   - **CRITICAL**: Processes ONE frame at a time to avoid memory explosion (not true batching)
   - For each frame triplet:
     - Apply multi-image chat template with 3 images
     - Shows "Frame BEFORE", "Frame CURRENT", "Frame AFTER"
     - Prompt focuses on ACTION and MOVEMENT detection
     - Generate description with max 128 new tokens
     - Explicitly delete tensors after each frame for memory efficiency
   - **Batch embed descriptions**: After processing N frames, encode all descriptions using sentence-transformers
   - Store 384-dimensional embedding vectors alongside text
7. **Incremental Saving & Output Generation** (embeddings/embed.py:189-204):
   - Save JSON checkpoint every 8 processed frames (default `save_interval`)
   - Results auto-sorted by frame number before each save
   - Intermediate saves marked with `"status": "in_progress"`
   - Final save marked with `"status": "complete"`
   - Output file naming:
     - Uploaded videos: `video_frames_analysis/{video_basename}.json`
     - YouTube videos: `video_frames_analysis/yt_{url_hash}.json`
   - Format: `{video_name, total_frames, frames: [{frame_number, timestamp, text, embedding}], status}`

8. **Search Interface** (shown if JSON exists OR after analysis completes):
   - Query input box for text search
   - Top-k slider (1-20 results, default: 5)
   - Padding configuration (0-5 seconds, default: 2s)
   - Loads embedding model and computes cosine similarity (embeddings/query.py:31-57)
   - **Clip extraction modes**:
     - If video file available: Extract clips with FFmpeg (stream copy or re-encode fallback)
     - If embeddings-only mode (YouTube without download): Show results without clips
   - Displays results with:
     - Similarity score (cosine similarity)
     - Frame description
     - Timestamp (from stored metadata, not calculated)
     - Extracted video clip with padding (if video available)
     - Download button for each clip
   - Saves clips to `extracted_clips/` directory

### Key Components

**Model Loading (embeddings/load.py)**
- `load_model()` (lines 16-35): Caches VLM and processor via `@st.cache_resource`
  - Uses `AutoModelForImageTextToText` and `AutoProcessor` from transformers
  - Fixed to CPU device with bfloat16 precision
  - Avoids `device_map="auto"` to prevent meta device errors
- `load_embedding_model()` (lines 38-42): Caches sentence-transformers model

**Temporal Frame Extraction (embeddings/embed.py:20-83)**
- Two-pass extraction for temporal context:
  1. First pass: Load ALL video frames into memory buffer (lines 38-46)
  2. Second pass: For each sampled position, extract 3-frame windows (lines 56-81)
- Opens video with OpenCV (`cv2.VideoCapture`)
- Extracts FPS and total frame count
- Samples frames at time-based interval (default 1.5 seconds)
- Creates temporal triplets with time-based offsets (±0.8 seconds)
- Returns frame data with `frame_number`, `actual_frame_index`, `timestamp`, and `images` (3 PIL Images)

**Frame-by-Frame Processing (embeddings/embed.py:86-186)**
- **Memory optimization**: Processes ONE frame at a time, not true batching
- Split frames into batches for checkpoint frequency only
- For each frame triplet:
  - Build multi-image conversation with 3 images (lines 103-116)
  - Apply chat template and generate with VLM (lines 124-152)
  - Explicitly delete tensors after processing (lines 163-176)
- Batch embed all descriptions after processing batch (lines 178-184)
- Progress tracking updates after each batch

**Incremental Saving (embeddings/embed.py:189-204)**
- Auto-saves progress every N frames (configurable via `save_interval`)
- Prevents data loss if processing is interrupted
- Each save includes all processed frames sorted by frame number
- Status field: "in_progress" or "complete"
- File naming: uses video basename or YouTube URL hash
- UI shows "💾 Saved checkpoint" message when saving occurs

**FFmpeg Clip Extraction (embeddings/query.py:79-171)**
- Primary method: Stream copy (`-c copy`) for instant extraction
- Fallback: Re-encode with H.264 if stream copy fails
- Windows compatibility: `CREATE_NO_WINDOW` flag to hide FFmpeg console
- Validates output file exists and has non-zero size
- Error handling with Streamlit UI integration

**Multi-Image Conversation Template (embeddings/embed.py:103-116)**
- Uses vision-language chat format with role-based messages
- Conversation structure:
  - "Frame BEFORE:" + image
  - "Frame CURRENT (describe this one):" + image
  - "Frame AFTER:" + image
  - Action-focused prompt (line 92)
- Prompt designed for action/movement detection, not static descriptions

**Output Format**
```json
{
  "video_name": "example.mp4",
  "total_frames": 120,
  "status": "complete",
  "frames": [
    {
      "frame_number": 1,
      "timestamp": 0.0,
      "actual_frame_index": 0,
      "text": "person jumping to shoot basketball into hoop",
      "embedding": [0.123, -0.456, 0.789, ... ]
    },
    {
      "frame_number": 2,
      "timestamp": 1.5,
      "actual_frame_index": 45,
      "text": "player running down court with ball",
      "embedding": [-0.234, 0.567, -0.890, ... ]
    }
  ]
}
```

**Field descriptions:**
- `status`: "in_progress" during processing, "complete" when finished
- `frame_number`: Sequential frame number (1, 2, 3, ...)
- `timestamp`: Time in seconds where this frame appears in video
- `actual_frame_index`: Actual frame index in video file (0-based)
- `text`: VLM-generated action description
- `embedding`: 384-dimensional vector for semantic search (cosine similarity)

## Important Implementation Details

### Frame Sampling Strategy
- **Time-based sampling** (not frame-based): Ensures consistent sampling across different FPS
- Default: Every 1.5 seconds (`sample_interval_seconds` in embeddings/embed.py:217)
  - 30fps video: ~45 frames between samples
  - 60fps video: ~90 frames between samples
- Temporal window offset: ±0.8 seconds (`temporal_offset_seconds` in embeddings/embed.py:218)
  - 30fps video: ±24 frames
  - 60fps video: ±48 frames
- Adjustable by modifying parameters in `process_video()` call
- Trade-off: Shorter interval = more detail but slower processing
- **Memory warning**: ALL frames loaded into memory before sampling

### Batch Processing Performance
- **IMPORTANT**: "Batch size" is misleading - frames are processed ONE AT A TIME
  - Batch size only controls checkpoint save frequency
  - True batching would cause memory explosion with multiple 3-image inputs
  - Memory optimization: explicit tensor deletion after each frame (embeddings/embed.py:163-176)
- **Worker threads**: Currently unused (UI setting has no effect)
  - Processing is sequential, not parallel
  - Setting displayed for reference only
- **Processing mode**: Frame-by-frame within batches, batches processed sequentially
- **Expected performance**: ~5-15 sec/frame on CPU
- **Checkpoint frequency**: Set `save_interval` parameter in `process_video()` (default: 8 frames)

### Model Prompt Engineering
- Current prompt focuses on action and movement detection (embeddings/embed.py:92)
- Full prompt: "You are viewing 3 consecutive frames from a video (before, current, after). Describe what ACTION or MOVEMENT is occurring in the middle frame. Focus on: 1) What the main subjects are DOING (not just their appearance), 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."
- Can be customized in `process_frame_batch()` for different use cases:
  - Static descriptions: "Describe the objects and scene in the middle frame"
  - Object tracking: "Identify which objects moved between frames"
  - Scene transitions: "Describe how the scene changed from before to after"

### Performance Considerations
- **First run**: Model download from HuggingFace (~3.2GB for LFM2-VL-1.6B + ~90MB for embedding model)
- **GPU NOT supported**: Currently hardcoded to CPU device
- **Memory usage**:
  - VLM requires ~4GB RAM minimum
  - Embedding model: ~90MB
  - Video frames: ALL frames loaded into memory (can be large for long videos)
  - Each sampled frame creates 3 PIL Images (triplet)
  - Embeddings add ~1.5KB per frame (384 floats × 4 bytes)
  - **Critical**: One-frame-at-a-time processing prevents tensor memory explosion
- **Caching**: Streamlit caches both models via `@st.cache_resource` (persists between reruns, not sessions)
- **Device handling**: Fixed to CPU with bfloat16 precision (no automatic detection)
- **Processing time**: Expect 5-15 seconds per frame triplet on CPU
- **FFmpeg requirement**: Must be installed separately for clip extraction
  - Download from: https://ffmpeg.org/download.html
  - Add to system PATH

### Semantic Search Implementation

**How it Works (embeddings/query.py)**
1. Load the JSON file containing frame descriptions and embeddings
   - Uploaded videos: `{video_basename}.json`
   - YouTube videos: `yt_{url_hash}.json`
2. Load the same sentence-transformers model used during analysis
3. Embed the query text (e.g., "layup") into a 384-dimensional vector
4. Compute cosine similarity between query embedding and all frame embeddings (lines 31-57)
5. Return top-k frames sorted by similarity score (highest first)

**Embedding Model Choice**
- Uses `sentence-transformers/all-MiniLM-L6-v2` by default
- Lightweight (90MB), fast, good semantic understanding
- 384-dimensional embeddings balance quality and storage
- **IMPORTANT**: Query must use the same model as analysis to ensure embeddings are in the same vector space

**Search Quality**
- Works well for action/movement queries matching the prompt style
- Examples of good queries: "layup", "person jumping", "basketball shot", "running with ball"
- Text-to-text matching: Query is compared to generated descriptions, not raw images
- Advantages: Fast, efficient, works on CPU
- Limitations: Search quality depends on VLM's description quality

**Alternative Approaches (Not Implemented)**
- CLIP-based search: Use CLIP to embed both frames and queries directly (better cross-modal alignment)
- Hybrid search: Combine text similarity with temporal/spatial features
- Query expansion: Use VLM to expand short queries into detailed descriptions

## Extension Points for Future Development

The basic semantic search is now implemented. Additional features that could be added:

### 1. Vector Database Integration
- ✅ **DONE**: Text embeddings stored in JSON
- 🔄 **Enhancement**: Move to ChromaDB, Pinecone, or FAISS for faster similarity search
- Benefits: Faster queries, better scaling to multiple videos, filtering capabilities

### 2. Multi-Video Search
- Index embeddings from multiple videos
- Search across entire video library
- Video-level and frame-level results

### 3. Advanced Search Features
- Frame-to-frame: Upload reference image → find similar frames using CLIP
- Temporal search: Find sequences matching criteria (e.g., "shot followed by celebration")
- Filters: Search within specific videos, time ranges, or frame types

### 4. Better Result Presentation
- ✅ **DONE**: CLI shows top-k results with similarity scores
- 🔄 **Enhancement**: Web UI for search results
- Display matching frames as thumbnails with timestamps
- Generate video clips around matches
- Timeline view showing match distribution

## Current Limitations

- **Single video search**: Cannot search across multiple videos at once
- **No vector database**: Embeddings stored in JSON, slow for large datasets
- **Text-based search only**: No image-to-frame search capability
- **GPU not utilized**: Hardcoded to CPU device, no CUDA support
- **Fixed prompting**: Single hardcoded prompt for all frames (customizable but requires code changes)
- **Memory intensive**: ALL video frames loaded into memory at once
- **Worker setting unused**: UI shows worker threads but processing is sequential (one frame at a time)
- **Single video at a time**: No batch video processing
- **FFmpeg dependency**: Requires separate installation for clip extraction
- **YouTube mode limitation**: Can search embeddings without video file, but cannot extract clips

## Key Architectural Decisions

Important design choices and their rationale:

1. **Modular Package Structure**: Code organized into `embeddings/` package
   - `load.py`: Model loading and caching
   - `embed.py`: Video processing and embedding generation
   - `query.py`: Search and clip extraction
   - Improves maintainability and testability

2. **Time-Based Sampling**: Samples every 1.5 seconds (not every N frames)
   - Ensures consistent sampling across different FPS videos
   - More intuitive for users to understand temporal coverage

3. **One-Frame-At-A-Time Processing**: Despite "batch size" parameter
   - Prevents memory explosion from multiple 3-image inputs
   - Explicit tensor deletion after each frame (lines 163-176 in embed.py)
   - Batch size only controls checkpoint frequency

4. **YouTube URL Hashing**: Uses SHA256 hash for caching
   - Enables searching existing embeddings without re-downloading video
   - Useful for repeated searches on same YouTube content

5. **FFmpeg for Clip Extraction**: Replaced OpenCV-based approach
   - Stream copy mode for instant extraction (no re-encoding)
   - Fallback to re-encode if stream copy fails
   - Much faster than frame-by-frame writing

6. **Temporal Context Windows**: 3-frame analysis (before, current, after)
   - Enables action/movement detection (not just static descriptions)
   - ±0.8 second offset provides sufficient temporal context

7. **Incremental Saving**: Checkpoints every 8 frames by default
   - Prevents data loss during long processing sessions
   - Allows resuming from partial results (status: "in_progress")

8. **Stored Timestamps**: JSON includes actual timestamps, not just frame numbers
   - Eliminates need to recalculate from FPS during search
   - More accurate for variable frame rate videos

## Common Development Tasks

### Adjusting Frame Sampling Rate
Edit `process_video()` call in main.py:407:
```python
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
    save_interval=8
)
```

Then modify `extract_frames()` call in embeddings/embed.py:215-219:
```python
frames_data, fps, total_frames = extract_frames(
    video_path,
    sample_interval_seconds=1.5,  # Change this: lower = more frames, slower
    temporal_offset_seconds=0.8   # Change this: temporal context window
)
```

### Changing the VLM Prompt
Edit the prompt in embeddings/embed.py:92:
```python
prompt = "Your custom prompt here..."
```

### Using Different Models
- **VLM**: Change `model_id` in embeddings/load.py:16
- **Embedding model**: Change `model_id` in embeddings/load.py:39
- **Important**: If changing embedding model, re-process all videos (embeddings incompatible)

### Testing Search Without Video Processing
If you have an existing JSON file:
```bash
python -m embeddings.query "your query" --json-file path/to/video_analysis.json --top-k 10
```

### Troubleshooting

**"FFmpeg not found" error:**
- Install FFmpeg: https://ffmpeg.org/download.html
- Windows: Add FFmpeg `bin/` folder to system PATH
- Verify: `ffmpeg -version` in terminal

**Memory errors during processing:**
- Reduce video length (clip video first)
- Lower frame sampling rate (increase `sample_interval_seconds`)
- Process is already one-frame-at-a-time; further optimization requires code changes

**Model download issues:**
- Check internet connection
- Models download from HuggingFace automatically on first run
- ~3.2GB for VLM + ~90MB for embedding model
- Cached in `~/.cache/huggingface/` (Linux/Mac) or `C:\Users\<user>\.cache\huggingface\` (Windows)

**Search returns poor results:**
- Ensure query matches the action-focused prompt style
- Good: "person jumping", "layup", "running with ball"
- Poor: "basketball", "court", "jersey number 23"
- VLM quality affects search quality - descriptions must be accurate

**YouTube download fails:**
- Check URL is valid and video is accessible
- yt-dlp may need updating: `pip install -U yt-dlp`
- Some videos may be region-restricted or require authentication
