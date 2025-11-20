"""
Model loading utilities for video-language model and embedding model.

This module handles:
- GPU/CPU-based model loading with auto-detection
- Model caching for performance
"""

import torch
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer


def detect_device():
    """
    Detect the best available device for model inference.

    Returns:
        tuple: (device_type, device_name) where device_type is "cuda" or "cpu"
               and device_name is a human-readable string
    """
    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        return device_type, device_name
    else:
        device_type = "cpu"
        device_name = "CPU"
        return device_type, device_name


@st.cache_resource
def load_model(model_id="LiquidAI/LFM2-VL-1.6B", device_type="auto"):
    """
    Load the vision-language model and processor.

    Args:
        model_id: HuggingFace model identifier
        device_type: "auto" (auto-detect), "cuda", or "cpu"

    Returns:
        tuple: (model, processor, config_dict)
    """
    # Auto-detect device if requested
    if device_type == "auto":
        device_type, _ = detect_device()

    # Determine dtype based on device
    if device_type == "cuda":
        torch_dtype = torch.float16  # Use float16 for GPU efficiency
        dtype_name = "float16"
    else:
        torch_dtype = torch.bfloat16  # Use bfloat16 for CPU
        dtype_name = "bfloat16"

    # Configuration dictionary
    config = {
        "device": device_type,
        "dtype": dtype_name,
        "torch_dtype": torch_dtype
    }

    # Load model with appropriate configuration
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=config["torch_dtype"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Move model to target device
    model = model.to(device_type)

    return model, processor, config


@st.cache_resource
def load_embedding_model(model_id="sentence-transformers/all-MiniLM-L6-v2", device_type="auto"):
    """
    Load the sentence transformer model for embedding text descriptions.

    Args:
        model_id: HuggingFace model identifier
        device_type: "auto" (auto-detect), "cuda", or "cpu"

    Returns:
        SentenceTransformer model
    """
    # Auto-detect device if requested
    if device_type == "auto":
        device_type, _ = detect_device()

    # SentenceTransformer accepts device string directly
    embedding_model = SentenceTransformer(model_id, device=device_type)
    return embedding_model
