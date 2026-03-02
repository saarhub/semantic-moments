"""DINOv2 embedder for SemanticMoments."""

import torch
import torchvision.transforms as T

from .base import Embedder
from ..utils import sample_frames_uniformly


class DINOEmbedder(Embedder):
    """SemanticMoments embedder using DINOv2 backbone.

    Extracts patch features from DINOv2 and computes temporal moments.

    Args:
        model_name: DINOv2 model variant. Options:
            - "dinov2_vits14": ViT-S/14 (384 dim)
            - "dinov2_vitb14": ViT-B/14 (768 dim)
            - "dinov2_vitl14": ViT-L/14 (1024 dim)
            - "dinov2_vitg14": ViT-G/14 (1536 dim)
            - "dinov2_vitl14_reg": ViT-L/14 with registers (1024 dim)
            - "dinov2_vitg14_reg": ViT-G/14 with registers (1536 dim)
        num_frames: Number of frames to sample. Default: 32
        alpha1: Weight for first moment. Default: 1.0
        alpha2: Weight for second moment. Default: 8.0
        alpha3: Weight for third moment. Default: 4.0
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        num_frames: int = 32,
        alpha1: float = 1.0,
        alpha2: float = 8.0,
        alpha3: float = 4.0,
        aggregation: str = "concat",
    ):
        super().__init__(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, aggregation=aggregation)
        self.model_name = model_name
        self.num_frames = num_frames

        # Load DINOv2 model
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(self.device).eval()

        # ImageNet normalization
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def embed_video(self, video_frames: list) -> torch.Tensor:
        """Embed video using DINOv2 patch features.

        Args:
            video_frames: List of PIL Images.

        Returns:
            Video embedding tensor.
        """
        # Sample frames uniformly
        video_frames = sample_frames_uniformly(video_frames, num_frames=self.num_frames)

        # Transform and stack frames
        frames_tensor = torch.stack([self.transform(f) for f in video_frames])
        frames_tensor = frames_tensor.to(self.device)

        # Extract patch features
        with torch.no_grad():
            features = self.model.forward_features(frames_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (T, P, D)

        # Compute moments
        return self.compute_moments(patch_tokens)
