#!/usr/bin/env python3
"""
Semantic Video Search Script

This script searches through video frame embeddings to find frames
matching a text query (e.g., "layup", "person jumping", etc.)

Usage:
    python search.py "your query here"
    python search.py "layup" --top-k 5
    python search.py "person running" --json-file path/to/video_analysis.json
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_frames(query, json_path, top_k=5, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Search for frames matching the query

    Args:
        query: Text query (e.g., "layup", "person jumping")
        json_path: Path to video_analysis.json file
        top_k: Number of top results to return
        model_id: Sentence transformer model to use (must match the one used for embedding)

    Returns:
        List of (frame_number, text, similarity_score) tuples
    """
    # Load the analysis JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if embeddings exist
    if not data['frames'] or 'embedding' not in data['frames'][0]:
        raise ValueError("No embeddings found in JSON file. Please re-run the analysis with the updated code.")

    # Load the embedding model (same one used during analysis)
    print(f"Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id)

    # Embed the query
    print(f"Embedding query: '{query}'")
    query_embedding = model.encode(query, convert_to_numpy=True)

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


def main():
    parser = argparse.ArgumentParser(
        description="Search video frames by semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search.py "layup"
  python search.py "person jumping" --top-k 10
  python search.py "basketball shot" --json-file custom_analysis.json
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

    # Perform search
    try:
        results = search_frames(
            query=args.query,
            json_path=args.json_file,
            top_k=args.top_k,
            model_id=args.model
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
        return


if __name__ == "__main__":
    main()
