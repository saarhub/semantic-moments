"""VideoMAE embedder for SemanticMoments."""

import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from .base import Embedder
from ..utils import sample_frames_uniformly


class VideoMAEEmbedder(Embedder):
    """SemanticMoments embedder using VideoMAE backbone.

    Extracts patch features from VideoMAE and computes temporal moments.

    Args:
        model_name: HuggingFace model name. Default: "MCG-NJU/videomae-base-finetuned-kinetics"
        num_frames: Number of frames to sample. Default: 16
        alpha1: Weight for first moment. Default: 1.0
        alpha2: Weight for second moment. Default: 8.0
        alpha3: Weight for third moment. Default: 4.0
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        num_frames: int = 16,
        alpha1: float = 1.0,
        alpha2: float = 8.0,
        alpha3: float = 4.0,
        aggregation: str = "concat",
    ):
        super().__init__(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, aggregation=aggregation)
        self.model_name = model_name
        self.num_frames = num_frames

        # Load VideoMAE model and processor
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        self.model.to(self.device).eval()

        # VideoMAE config
        self.temporal_patches = 8  # VideoMAE uses 8 temporal tokens for 16 frames
        self.spatial_patches = 196  # 14x14 spatial patches
        self.hidden_dim = 768

    def embed_video(self, video_frames: list) -> torch.Tensor:
        """Embed video using VideoMAE patch features.

        Args:
            video_frames: List of PIL Images or numpy arrays.

        Returns:
            Video embedding tensor.
        """
        # Sample frames uniformly
        video_frames = sample_frames_uniformly(video_frames, num_frames=self.num_frames)

        # Process frames
        inputs = self.processor(video_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features from base model
        with torch.no_grad():
            outputs = self.model.base_model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, T*P, D)

            # Reshape to (T, P, D)
            patch_tokens = hidden_states.view(
                self.temporal_patches,
                self.spatial_patches,
                self.hidden_dim
            )

        # Compute moments
        return self.compute_moments(patch_tokens)
