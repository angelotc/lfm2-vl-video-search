"""
Embeddings package for semantic video search.

This package provides modular components for:
- CPU-based model loading (VLM and embedding models)
- Video processing and embedding generation
- Semantic search and query functionality
"""

from .load import (
    load_model,
    load_embedding_model
)

from .embed import (
    extract_frames,
    process_frame_batch,
    process_video,
    save_incremental_json
)

from .query import (
    cosine_similarity,
    search_frames,
    get_frame_timestamp,
    extract_clip,
    main as query_main
)

__all__ = [
    # Model loading
    'load_model',
    'load_embedding_model',

    # Embedding generation
    'extract_frames',
    'process_frame_batch',
    'process_video',
    'save_incremental_json',

    # Search/query
    'cosine_similarity',
    'search_frames',
    'get_frame_timestamp',
    'extract_clip',
    'query_main',
]
