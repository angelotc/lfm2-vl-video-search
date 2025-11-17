# Semantic Video Search with LFM2-VL

A Python-based semantic video search system using LiquidAI's LFM2-VL-1.6B vision-language model. Search through videos using natural language queries to find specific actions, movements, and events.

## Features

### Core Capabilities
- **Semantic Video Search**: Find frames by describing what you're looking for in natural language
- **Temporal Frame Analysis**: 3-frame context windows (before/current/after) for action detection
- **Multiple Input Methods**: Upload video files or provide YouTube URLs
- **Smart Caching**: Automatically detects and reuses existing analysis files
- **Clip Extraction**: Generate video clips around search results with configurable padding
- **Vector Embeddings**: 384-dimensional semantic embeddings for fast similarity search

### Supported Video Formats
- MP4, AVI, MOV, MKV (via file upload or YouTube download)

## Installation

### Prerequisites
- **Python 3.8+**
- **FFmpeg** (required for clip extraction)
  - Download from: https://ffmpeg.org/download.html
  - Add to system PATH

### Setup

1. Create a virtual environment (recommended):
```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Main Application (Recommended)

Run the full-featured semantic video search app:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

#### Workflow:
1. **Choose Input Method**:
   - Upload a video file (MP4, AVI, MOV, MKV)
   - OR enter a YouTube URL to download and analyze

2. **Auto-Detection**: App checks for existing analysis
   - Uploaded videos: looks for `{video_basename}.json`
   - YouTube videos: looks for `yt_{url_hash}.json`
   - If found: Skip to search interface immediately
   - If not found: Process video with VLM

3. **Video Analysis** (if needed):
   - Configure batch size (1-16 frames, default: 4)
   - Process video with temporal frame analysis
   - Generate semantic descriptions using LFM2-VL-1.6B
   - Create vector embeddings for each frame
   - Progress saved incrementally

4. **Search Interface**:
   - Enter natural language query (e.g., "person jumping", "layup", "running with ball")
   - Adjust number of results (top-k: 1-20)
   - Configure clip padding (0-5 seconds)
   - View results with similarity scores
   - Download extracted clips

### Simple Upload App

For basic video upload and preview without AI processing:
```bash
streamlit run app.py
```


## Project Structure

```
.
├── main.py                     # Full-featured semantic search application
├── app.py                      # Simple video upload UI (no AI)
├── requirements.txt            # Python dependencies
├── embeddings/                 # Core processing modules
│   ├── load.py                # Model loading and caching
│   ├── embed.py               # Video processing and embedding generation
│   └── query.py               # Search and clip extraction
├── uploads/                    # Uploaded videos (auto-created)
├── video_frames_analysis/      # Analysis JSON files with embeddings
│   ├── {video_name}.json      # Uploaded video analysis
│   └── yt_{hash}.json         # YouTube video analysis
├── extracted_clips/            # Search result clips (auto-created)
└── README.md                   # This file
```

## How It Works

### 1. Temporal Frame Extraction
- Samples frames every 1.5 seconds (configurable)
- Creates 3-frame temporal windows (±0.8 seconds)
- Provides context for action/movement detection

### 2. Vision-Language Model Processing
- Uses LiquidAI LFM2-VL-1.6B model
- Analyzes each frame triplet with action-focused prompt
- Generates semantic descriptions of movements and actions
- Processes one frame at a time for memory efficiency

### 3. Embedding Generation
- Converts descriptions to 384-dimensional vectors
- Uses sentence-transformers/all-MiniLM-L6-v2 model
- Enables fast semantic similarity search

### 4. Search & Clip Extraction
- Computes cosine similarity between query and frame embeddings
- Ranks frames by relevance
- Extracts video clips using FFmpeg (stream copy or re-encode)
- Adds configurable padding around matching frames

## Performance Notes

- **First run**: Downloads ~3.2GB VLM model + ~90MB embedding model
- **Processing speed**: ~5-15 seconds per frame on CPU
- **Memory usage**: Requires ~4GB RAM minimum
- **Device**: CPU or GPU (check out the `Only-works-on-AMD` branch for ROCm support via llama.cpp)
- **Caching**: Models cached between runs, analysis files persisted

## Output Format

Analysis files are saved as JSON:

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
      "embedding": [0.123, -0.456, ...]
    }
  ]
}
```

## Configuration

### Sampling Parameters
- **Frame interval**: 1.5 seconds (modify in `embeddings/embed.py`)
- **Temporal window**: ±0.8 seconds (modify in `embeddings/embed.py`)
- **Batch size**: Controls checkpoint frequency (UI configurable)

### Search Parameters
- **Top-k results**: 1-20 (UI configurable)
- **Clip padding**: 0-5 seconds (UI configurable)

## Limitations

- Single video search (no multi-video indexing yet)
- No vector database (embeddings stored in JSON)
- CPU-only processing (GPU support planned)
- Text-based search only (no image-to-frame search)
- Requires separate FFmpeg installation

## Future Enhancements

- Vector database integration (ChromaDB, FAISS)
- Multi-video search across entire library
- GPU acceleration support
- Image-to-frame search using CLIP
- Temporal sequence search
- Web-based result visualization

## Troubleshooting

**FFmpeg not found**: Ensure FFmpeg is installed and added to system PATH

**Out of memory**: Reduce batch size or use shorter videos

**Slow processing**: Expected on CPU; consider using shorter sample intervals for faster processing (fewer frames)

**Model download fails**: Check internet connection and HuggingFace access

## License

See LICENSE file for details.
