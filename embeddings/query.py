#!/usr/bin/env python3
"""
Search and query utilities for semantic video search.

This module handles:
- Cosine similarity computation
- Semantic frame search
- Timestamp lookup
- Video clip extraction
- CLI interface for searching videos

Usage (as CLI):
    python -m embeddings.query "your query here"
    python -m embeddings.query "layup" --top-k 5
    python -m embeddings.query "person running" --json-file path/to/video_analysis.json
"""

import json
import os
import subprocess
import argparse
import numpy as np
from pathlib import Path


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


def get_frame_timestamp(frame_number, json_data):
    """Calculate timestamp for a frame number from stored data

    Args:
        frame_number: The frame number to look up
        json_data: The loaded JSON data containing frame information with timestamps

    Returns:
        The timestamp in seconds for the given frame number
    """
    # Find the frame in the JSON data
    for frame in json_data['frames']:
        if frame['frame_number'] == frame_number:
            return frame.get('timestamp', 0.0)

    # Fallback: if not found, return 0
    return 0.0


def extract_clip(video_path, start_time, end_time, output_path, show_warnings=True):
    """Extract a clip from video between start_time and end_time (in seconds) using FFmpeg

    Args:
        video_path: Path to input video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save the clip
        show_warnings: If True, display warnings using streamlit (requires streamlit context)

    Returns:
        True if successful, False otherwise
    """
    try:
        duration = end_time - start_time

        # Use FFmpeg for fast, reliable clip extraction
        # -ss before -i = fast seeking (input seeking)
        # -c copy = stream copy (no re-encoding, instant extraction)
        # -avoid_negative_ts make_zero = handle timestamp edge cases
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-ss', str(start_time),  # Start time
            '-i', video_path,  # Input file
            '-t', str(duration),  # Duration
            '-c', 'copy',  # Stream copy (no re-encoding)
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0  # Hide window on Windows
        )

        # Check if extraction was successful
        if result.returncode != 0:
            # If stream copy failed, try re-encoding
            if show_warnings:
                import streamlit as st
                st.warning(f"Stream copy failed, trying re-encode...")

            cmd_reencode = [
                'ffmpeg',
                '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',  # Re-encode with H.264
                '-preset', 'ultrafast',  # Fast encoding
                '-c:a', 'aac',  # Re-encode audio
                output_path
            ]

            result = subprocess.run(
                cmd_reencode,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            if result.returncode != 0:
                if show_warnings:
                    import streamlit as st
                    st.error(f"FFmpeg error: {result.stderr}")
                return False

        # Verify file was created and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if show_warnings:
                import streamlit as st
                st.error(f"Clip file not created or empty: {output_path}")
            return False

        return True

    except FileNotFoundError:
        if show_warnings:
            import streamlit as st
            st.error("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
            st.info("Download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        if show_warnings:
            import streamlit as st
            st.error(f"Error extracting clip: {e}")
            import traceback
            st.error(traceback.format_exc())
        return False


# CLI Interface
def main():
    """Command-line interface for semantic video search"""
    parser = argparse.ArgumentParser(
        description="Search video frames by semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m embeddings.query "layup"
  python -m embeddings.query "person jumping" --top-k 10
  python -m embeddings.query "basketball shot" --json-file custom_analysis.json

  # For backwards compatibility, you can also use:
  python search.py "layup"
        """
    )

    parser.add_argument(
        'query',
        type=str,
        help='Text query to search for (e.g., "layup", "person running")'
    )

    parser.add_argument(
        '--json-file',
        type=str,
        default='video_frames_analysis/video_analysis.json',
        help='Path to the video analysis JSON file (default: video_frames_analysis/video_analysis.json)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model ID (default: sentence-transformers/all-MiniLM-L6-v2)'
    )

    args = parser.parse_args()

    # Check if JSON file exists
    if not Path(args.json_file).exists():
        print(f"Error: JSON file not found: {args.json_file}")
        print("Please run the video analysis first using main.py")
        return

    # Load the embedding model
    print(f"Loading embedding model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # Perform search
    try:
        print(f"Embedding query: '{args.query}'")
        results = search_frames(
            query=args.query,
            json_path=args.json_file,
            embedding_model=model,
            top_k=args.top_k
        )

        # Display results
        print(f"\n{'='*80}")
        print(f"Top {len(results)} results for query: '{args.query}'")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Frame {result['frame_number']} (similarity: {result['similarity']:.4f})")
            print(f"   Description: {result['text']}")
            print()

        # Show video name
        with open(args.json_file, 'r') as f:
            data = json.load(f)
            print(f"Video: {data['video_name']}")
            print(f"Total frames analyzed: {data['total_frames']}")

    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
