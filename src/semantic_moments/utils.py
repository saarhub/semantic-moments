"""Utility functions for SemanticMoments."""

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm


def sample_frames_uniformly(frames: list, num_frames: int = 32) -> list:
    """Sample frames uniformly from a video.

    Args:
        frames: List of video frames (PIL Images or tensors).
        num_frames: Number of frames to sample.

    Returns:
        List of sampled frames.
    """
    total = len(frames)
    if total <= num_frames:
        return frames
    indices = torch.linspace(0, total - 1, steps=num_frames).long()
    return [frames[i] for i in indices]


def _uniform_indices(total: int, num_frames: int) -> list:
    """Compute uniform frame indices."""
    if num_frames <= 1:
        return [0] if total > 0 else []
    return [int(round(t)) for t in np.linspace(0, total - 1, num_frames)]


def load_video_frames(
    video_path: str,
    num_frames: int = 32,
    target_size: tuple = (224, 224),
    return_pil: bool = True,
) -> list:
    """Load video frames uniformly sampled from a video file.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample uniformly.
        target_size: (height, width) to resize frames.
        return_pil: If True, return PIL Images. If False, return numpy arrays.

    Returns:
        List of frames (PIL Images or numpy arrays RGB uint8).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    indices = _uniform_indices(total, num_frames)
    frames = []
    next_idx = 0
    target = indices[next_idx]

    i = 0
    ok, frame = cap.read()
    while ok and next_idx < len(indices):
        if i == target:
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            if return_pil:
                frame = Image.fromarray(frame)
            frames.append(frame)
            next_idx += 1
            if next_idx < len(indices):
                target = indices[next_idx]
            else:
                break
        ok, frame = cap.read()
        i += 1

    cap.release()
    return frames


def embed_videos(
    embedder,
    video_paths: list,
    num_frames: int = None,
    target_size: tuple = (224, 224),
    show_progress: bool = True,
) -> torch.Tensor:
    """Embed multiple videos.

    Args:
        embedder: An Embedder instance.
        video_paths: List of paths to video files.
        num_frames: Number of frames to load per video. If None, uses embedder.num_frames.
        target_size: (height, width) to resize frames.
        show_progress: Show progress bar.

    Returns:
        Tensor of shape (N, D) where N is number of videos and D is embedding dim.
    """
    if num_frames is None:
        num_frames = getattr(embedder, "num_frames", 32)

    embeddings = []
    iterator = tqdm(video_paths, desc="Embedding videos") if show_progress else video_paths

    for video_path in iterator:
        frames = load_video_frames(
            video_path,
            num_frames=num_frames,
            target_size=target_size,
            return_pil=True,
        )
        emb = embedder.embed_video(frames)
        embeddings.append(emb)

    return torch.stack(embeddings)


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between embeddings.

    Args:
        embeddings: Tensor of shape (N, D), assumed L2-normalized.

    Returns:
        Similarity matrix of shape (N, N).
    """
    return embeddings @ embeddings.T
