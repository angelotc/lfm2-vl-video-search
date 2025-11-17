"""
Model loading utilities for video-language model and embedding model.

This module handles:
- CPU-based model loading
- Model caching for performance
"""

import torch
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model(model_id="LiquidAI/LFM2-VL-1.6B", device_type="cpu"):
    """Load the vision-language model and processor (CPU only)"""
    # CPU-only configuration
    config = {
        "device": "cpu",
        "dtype": "bfloat16",
        "torch_dtype": torch.bfloat16
    }

    # Load model with CPU configuration
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=config["torch_dtype"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Move model to CPU
    model = model.to("cpu")

    return model, processor, config


@st.cache_resource
def load_embedding_model(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence transformer model for embedding text descriptions"""
    embedding_model = SentenceTransformer(model_id)
    return embedding_model
