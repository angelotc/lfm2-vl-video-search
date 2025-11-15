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
pip install streamlit transformers torch opencv-python pillow accelerate
```

## Architecture

### Project Structure
- **`app.py`**: Simple video upload and preview UI (no AI processing)
  - File uploader supporting MP4, AVI, MOV, MKV, WMV, FLV, WebM
  - Video preview and metadata display
  - Save videos to `uploads/` directory

- **`main.py`**: Core semantic video processing application
  - Frame extraction at 1 FPS intervals
  - VLM-based frame description generation
  - JSON output with frame-level annotations
  - Progress tracking and visualization

- **`uploads/`**: Directory for saved uploaded videos (auto-created)

- **`video_frames_analysis/`**: Output directory for analysis results (auto-created)

### Core Processing Flow (main.py)

1. **Video Upload**: User uploads video via Streamlit file uploader
2. **Performance Configuration**: User sets batch size and worker threads
   - Batch size: 1-16 frames per batch (default: 4)
   - Worker threads: 1-4 parallel workers (default: 2)
3. **Model Loading**: Cached loading of LFM2-VL model
   - Model: `LiquidAI/LFM2-VL-450M` from HuggingFace (configurable)
   - Device: CPU with float16 precision
   - Manual device management (no device_map="auto")
4. **Frame Extraction** (Phase 1):
   - Extract frames at fixed interval (every 30 frames) using OpenCV
   - Convert BGR to RGB for model compatibility
   - Store all frames in memory before processing
5. **Parallel VLM Processing** (Phase 2):
   - Split frames into batches
   - Process batches in parallel using ThreadPoolExecutor
   - Each worker processes one batch at a time
   - Apply chat template with image + text prompt
   - Prompt: "Describe this frame in 10 words or less using keywords only."
   - Generate description with max 128 new tokens
6. **Output Generation**:
   - Sort results by frame number
   - Save results to `video_frames_analysis/video_analysis.json`
   - Format: `{video_name, total_frames, frames: [{frame_number, text}]}`
   - Display results in-app and provide JSON download

### Key Components

**Model Loading (`@st.cache_resource`)**
- Caches model and processor across reruns for performance
- Uses `AutoModelForImageTextToText` and `AutoProcessor` from transformers
- Automatic device detection (CUDA if available, otherwise CPU)
- Uses float16 on GPU for efficiency, float32 on CPU for compatibility
- Avoids `device_map="auto"` to prevent meta device errors

**Frame Extraction (`extract_frames`)**
- Separated extraction phase for better performance
- Opens video with OpenCV (`cv2.VideoCapture`)
- Extracts FPS and total frame count
- Samples frames at fixed interval (configurable, default every 30 frames)
- Stores all frames in memory as PIL Images before processing

**Parallel Processing (`process_video` + `process_frame_batch`)**
- Uses `ThreadPoolExecutor` for concurrent batch processing
- Frames split into configurable batches (default: 4 frames/batch)
- Multiple workers process batches in parallel (default: 2 workers)
- Progress tracking updates as batches complete
- Results sorted by frame number after all processing completes
- Converts each frame to PIL Image for model input

**Conversation Template**
- Uses vision-language chat format with role-based messages
- Each frame analyzed independently (no cross-frame context)
- Text prompt designed for concise keyword extraction

**Output Format**
```json
{
  "video_name": "example.mp4",
  "total_frames": 120,
  "frames": [
    {"frame_number": 1, "text": "person walking, outdoor scene, daytime"},
    {"frame_number": 2, "text": "building, street, cars parked"}
  ]
}
```

## Important Implementation Details

### Frame Sampling Strategy
- Current: Every 30 frames (`frame_interval = 30` in main.py:108)
- For 30fps video: ~1 frame per second
- For 60fps video: ~2 frames per second
- Adjustable by modifying `frame_interval` in `extract_frames()` call
- Trade-off: Higher sampling = more detail but slower processing

### Parallel Processing Performance
- **Batch size**: Controls memory usage and GPU utilization
  - Smaller batches (1-2): Lower memory, more overhead
  - Larger batches (8-16): Higher memory, better throughput (if GPU available)
  - Default (4): Good balance for CPU processing
- **Worker threads**: Controls parallelism
  - CPU-only: Limited benefit beyond 2-3 workers (GIL + model serialization)
  - GPU: Can benefit from more workers if VRAM allows
  - Default (2): Conservative setting for CPU
- **Expected speedup**: 1.5-2x on CPU with 2 workers, depends on batch size and model

### Model Prompt Engineering
- Current prompt focuses on keyword extraction (10 words or less)
- Located in line 69 of main.py
- Can be customized for different use cases:
  - Detailed descriptions: "Describe this frame in detail"
  - Object detection: "List all objects visible in this frame"
  - Scene classification: "Classify the scene type"

### Performance Considerations
- **First run**: Model download from HuggingFace (~3.2GB)
- **GPU recommended**: Processing 1 frame/second video can take significant time on CPU
- **Memory usage**: Model requires ~4GB RAM minimum (more on CPU)
- **Caching**: Streamlit caches model between reruns but not between sessions
- **Device handling**: Automatically uses GPU (CUDA) if available, falls back to CPU with float32

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
- **No cross-frame context**: Each frame analyzed independently
- **Fixed prompting**: Single hardcoded prompt for all frames
- **No batch processing**: Videos processed sequentially, one at a time
- **Incomplete requirements.txt**: Missing several critical dependencies
