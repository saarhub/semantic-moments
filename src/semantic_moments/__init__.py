"""SemanticMoments: Training-Free Motion Similarity via Third Moment Features."""

from .embedders import DINOEmbedder, VideoMAEEmbedder, VJEPA2Embedder, Embedder
from .utils import (
    load_video_frames,
    embed_videos,
    compute_similarity_matrix,
    sample_frames_uniformly,
)
from .datasets import SimMotionSynthetic, SimMotionReal, Triplet, download_simmotion

__version__ = "0.1.0"

__all__ = [
    # Embedders
    "Embedder",
    "DINOEmbedder",
    "VideoMAEEmbedder",
    "VJEPA2Embedder",
    # Utilities
    "load_video_frames",
    "embed_videos",
    "compute_similarity_matrix",
    "sample_frames_uniformly",
    # Datasets (with .evaluate() methods)
    "SimMotionSynthetic",
    "SimMotionReal",
    "Triplet",
    "download_simmotion",
]
