# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based semantic video search system using LiquidAI's LFM2-VL-1.6B vision-language model. The project processes videos frame-by-frame, generates semantic descriptions using the VLM, and enables semantic search capabilities across video content.

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
streamlit run main.py  # Full video frame analyzer with LFM2-VL model
```

Both apps will open in the browser at `http://localhost:8501`

### Required Dependencies
The current `requirements.txt` is incomplete. Full dependencies needed:
```bash
pip install streamlit transformers torch opencv-python pillow accelerate numpy
```

## Architecture

### Project Structure
- **`app.py`**: Simple video upload and preview UI (no AI processing)
  - File uploader supporting MP4, AVI, MOV, MKV, WMV, FLV, WebM
  - Video preview and metadata display
  - Save videos to `uploads/` directory

- **`main.py`**: Core semantic video processing application
  - Temporal frame extraction (before, current, after frames)
  - VLM-based action/movement description generation
  - Sequential batch processing with incremental saves
  - JSON output with frame-level annotations
  - Progress tracking and visualization

- **`uploads/`**: Directory for saved uploaded videos (auto-created)

- **`video_frames_analysis/`**: Output directory for analysis results (auto-created)

### Core Processing Flow (main.py)

1. **Video Upload**: User uploads video via Streamlit file uploader
2. **Performance Configuration**: User sets batch size and worker threads
   - Batch size: 1-16 frames per batch (default: 4)
   - Worker threads: 1-4 (displayed for reference, currently uses sequential processing)
3. **Model Loading**: Cached loading of LFM2-VL model
   - Model: `LiquidAI/LFM2-VL-450M` from HuggingFace (configurable)
   - Device: CPU with bfloat16 precision
   - Manual device management (no device_map="auto")
4. **Temporal Frame Extraction** (Phase 1):
   - Extract ALL frames into memory buffer using OpenCV
   - For each sampled frame (every 30 frames), create temporal window:
     - Frame N-1: 15 frames before current (or earliest available)
     - Frame N: Current frame
     - Frame N+1: 15 frames after current (or latest available)
   - Convert BGR to RGB for model compatibility
   - Store as triplets of PIL Images
5. **Sequential VLM Processing** (Phase 2):
   - Split temporal frame triplets into batches
   - Process batches sequentially (not parallel)
   - For each triplet, apply multi-image chat template:
     - Shows "Frame BEFORE", "Frame CURRENT", "Frame AFTER"
     - Prompt focuses on ACTION and MOVEMENT detection
     - Model analyzes temporal context to identify motion
   - Generate description with max 128 new tokens
   - Extract only new tokens (not prompt) from output
6. **Incremental Saving & Output Generation**:
   - Save JSON checkpoint every 8 processed frames
   - Results auto-sorted by frame number before each save
   - Intermediate saves marked with `"status": "in_progress"`
   - Final save marked with `"status": "complete"`
   - Output file: `video_frames_analysis/video_analysis.json`
   - Format: `{video_name, total_frames, frames: [{frame_number, text}], status}`
   - Display results in-app and provide JSON download

### Key Components

**Model Loading (`@st.cache_resource` in main.py:12-25)**
- Caches model and processor across reruns for performance
- Uses `AutoModelForImageTextToText` and `AutoProcessor` from transformers
- Fixed to CPU device (no automatic device detection)
- Uses bfloat16 precision for efficiency
- Avoids `device_map="auto"` to prevent meta device errors

**Temporal Frame Extraction (`extract_frames` in main.py:27-78)**
- Two-pass extraction for temporal context:
  1. First pass: Load ALL video frames into memory buffer
  2. Second pass: For each sampled position, extract 3-frame windows
- Opens video with OpenCV (`cv2.VideoCapture`)
- Extracts FPS and total frame count
- Samples frames at fixed interval (default every 30 frames)
- Creates temporal triplets: [before (-15 frames), current, after (+15 frames)]
- Returns list of frame data with `frame_number` and `images` (3 PIL Images)

**Sequential Batch Processing (`process_video` + `process_frame_batch` in main.py:80-208)**
- Split temporal frame triplets into batches (default: 4 triplets/batch)
- Process batches sequentially (not parallel, despite UI showing worker setting)
- For each frame triplet in batch:
  - Build multi-image conversation with 3 images and temporal prompt
  - Apply chat template with all 3 images inline
  - Generate description focusing on actions/movements
  - Decode only generated tokens (exclude prompt from output)
- Progress tracking updates after each batch
- Results collected and sorted by frame number

**Incremental Saving (`save_incremental_json` in main.py:138-151)**
- Auto-saves progress every 8 frames (configurable via `save_interval` parameter)
- Prevents data loss if processing is interrupted
- Each save includes all processed frames sorted by frame number
- Status field indicates "in_progress" or "complete"
- Same file (`video_analysis.json`) updated with latest results
- UI shows "💾 Saved checkpoint" message when saving occurs

**Multi-Image Conversation Template**
- Uses vision-language chat format with role-based messages
- Each frame gets temporal context from adjacent frames
- Conversation structure (main.py:96-109):
  - "Frame BEFORE:" + image
  - "Frame CURRENT (describe this one):" + image
  - "Frame AFTER:" + image
  - Action-focused prompt
- Prompt designed for action/movement detection, not static descriptions

**Output Format**
```json
{
  "video_name": "example.mp4",
  "total_frames": 120,
  "status": "complete",
  "frames": [
    {"frame_number": 1, "text": "person walking, outdoor scene, daytime"},
    {"frame_number": 2, "text": "building, street, cars parked"}
  ]
}
```

The `status` field will be:
- `"in_progress"` during processing (incremental saves)
- `"complete"` when all frames are processed

## Important Implementation Details

### Frame Sampling Strategy
- Current: Every 30 frames (`frame_interval = 30` in main.py:160)
- For 30fps video: ~1 frame per second
- For 60fps video: ~2 frames per second
- Temporal window offset: ±15 frames from sampled position
- Adjustable by modifying `frame_interval` in `extract_frames()` call
- Trade-off: Higher sampling = more detail but slower processing
- **Memory warning**: ALL frames loaded into memory before sampling

### Batch Processing Performance
- **Batch size**: Controls how many frame triplets processed per iteration
  - Smaller batches (1-2): Lower memory, more frequent progress updates
  - Larger batches (8-16): Higher memory usage (3 images per frame triplet)
  - Default (4): Processes 12 images per batch (4 triplets × 3 images each)
- **Worker threads**: Currently unused (UI setting has no effect)
  - Processing is sequential, not parallel
  - Setting displayed for future parallelization
- **Processing mode**: Sequential batches (no ThreadPoolExecutor)
- **Expected performance**: Linear with number of frames, ~5-15 sec/frame on CPU

### Model Prompt Engineering
- Current prompt focuses on action and movement detection (main.py:93)
- Full prompt: "You are viewing 3 consecutive frames from a video (before, current, after). Describe what ACTION or MOVEMENT is occurring in the middle frame. Focus on: 1) What the main subjects are DOING (not just their appearance), 2) Any motion or change between frames, 3) Specific actions like jumping, throwing, catching, running, etc. Be concise and action-focused."
- Can be customized for different use cases:
  - Static descriptions: "Describe the objects and scene in the middle frame"
  - Object tracking: "Identify which objects moved between frames"
  - Scene transitions: "Describe how the scene changed from before to after"

### Performance Considerations
- **First run**: Model download from HuggingFace (~3.2GB for LFM2-VL-450M)
- **GPU NOT supported**: Currently hardcoded to CPU device
- **Memory usage**:
  - Model requires ~4GB RAM minimum
  - Video frames: ALL frames loaded into memory (can be large for long videos)
  - Each sampled frame creates 3 PIL Images (triplet)
- **Caching**: Streamlit caches model between reruns but not between sessions
- **Device handling**: Fixed to CPU with bfloat16 precision (no automatic detection)
- **Processing time**: Expect 5-15 seconds per frame triplet on CPU

## Extension Points for Semantic Search

To build a complete semantic search system, the following components are needed:

### 1. Embedding Generation
- Replace keyword descriptions with vector embeddings
- Use VLM's hidden states or separate embedding model (e.g., CLIP)
- Store frame embeddings in vector database

### 2. Vector Storage
- ChromaDB, Pinecone, or FAISS for similarity search
- Index: video_id + frame_number + embedding vector
- Metadata: timestamp, video_name, original text description

### 3. Search Interface
- Text-to-frame: Encode search query → find nearest embeddings
- Frame-to-frame: Upload reference image → find similar frames
- Temporal search: Find sequences matching criteria

### 4. Result Presentation
- Display matching frames with timestamps
- Generate video clips around matches
- Relevance scoring and ranking

## Current Limitations

- **No semantic search implemented**: Only generates descriptions, no search capability
- **No embeddings**: Text descriptions are not vectorized
- **No indexing**: Results stored as flat JSON files
- **GPU not utilized**: Hardcoded to CPU device, no CUDA support
- **Fixed prompting**: Single hardcoded prompt for all frames
- **Memory intensive**: ALL video frames loaded into memory at once
- **Worker setting unused**: UI shows worker threads but processing is sequential
- **Single video at a time**: No batch video processing
- **Incomplete requirements.txt**: Missing several critical dependencies (torch, opencv-python, pillow, accelerate, numpy)

## Key Architectural Differences from Initial Design

The codebase has evolved significantly. Important changes to be aware of:

1. **Temporal Context**: Changed from single-frame to 3-frame temporal windows
2. **Processing Mode**: Changed from parallel ThreadPoolExecutor to sequential batch processing
3. **Prompt Focus**: Changed from keyword extraction to action/movement detection
4. **Device Handling**: Changed from auto-detection to hardcoded CPU
5. **Memory Strategy**: ALL frames now buffered in memory (two-pass extraction)
