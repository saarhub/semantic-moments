"""Base embedder class for SemanticMoments."""

import torch
import torch.nn.functional as F


class Embedder:
    """Base class for SemanticMoments embedders.

    All embedders compute temporal statistics (moments) over patch features
    from pretrained models to capture motion information.

    Args:
        alpha1: Weight for first moment (mean). Default: 1.0
        alpha2: Weight for second moment (std). Default: 8.0
        alpha3: Weight for third moment (skewness). Default: 4.0
        aggregation: How to combine moments - "concat" or "sum". Default: "concat"
        eps: Small value for numerical stability. Default: 1e-6
    """

    def __init__(
        self,
        alpha1: float = 1.0,
        alpha2: float = 8.0,
        alpha3: float = 4.0,
        aggregation: str = "concat",
        eps: float = 1e-6,
    ):
        if aggregation not in ("concat", "sum"):
            raise ValueError(f"aggregation must be 'concat' or 'sum', got '{aggregation}'")
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.aggregation = aggregation
        self.eps = eps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_moments(self, video_embedding: torch.Tensor) -> torch.Tensor:
        """Compute temporal moments over patch features.

        Args:
            video_embedding: Tensor of shape (T, P, D) where
                T = temporal dimension (frames or temporal patches)
                P = spatial patches
                D = feature dimension

        Returns:
            Moment embedding of shape (3*D,) for concat or (D,) for sum
        """
        # First moment: mean over time
        mean = video_embedding.mean(dim=0)  # (P, D)

        # Second moment: std over time
        std = video_embedding.std(dim=0)  # (P, D)

        # Third moment: skewness over time
        centered = video_embedding - mean
        skew = (centered ** 3).mean(dim=0) / (std ** 3 + self.eps)  # (P, D)

        # Spatial aggregation (mean over patches) and normalize
        mean_pooled = F.normalize(mean.mean(dim=0), dim=0) * self.alpha1
        std_pooled = F.normalize(std.mean(dim=0), dim=0) * self.alpha2
        skew_pooled = F.normalize(skew.mean(dim=0), dim=0) * self.alpha3

        # Aggregate moments
        if self.aggregation == "concat":
            embedding = torch.cat([mean_pooled, std_pooled, skew_pooled])
        else:  # sum
            embedding = mean_pooled + std_pooled + skew_pooled

        # Final normalization
        embedding = F.normalize(embedding, dim=0)

        return embedding

    def embed_video(self, video_frames: list) -> torch.Tensor:
        """Embed a video using SemanticMoments.

        Args:
            video_frames: List of video frames (PIL Images).

        Returns:
            Video embedding tensor of shape (3*D,) for concat or (D,) for sum
        """
        raise NotImplementedError("Subclasses must implement embed_video")

    def __call__(self, video_frames: list) -> torch.Tensor:
        """Alias for embed_video."""
        return self.embed_video(video_frames)
