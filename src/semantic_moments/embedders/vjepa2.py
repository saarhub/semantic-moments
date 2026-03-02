"""V-JEPA2 embedder for SemanticMoments."""

import torch
from transformers import AutoModel, AutoVideoProcessor

from .base import Embedder
from ..utils import sample_frames_uniformly


class VJEPA2Embedder(Embedder):
    """SemanticMoments embedder using V-JEPA2 backbone.

    Extracts patch features from V-JEPA2 and computes temporal moments.

    Args:
        model_size: Model variant. Options: "large", "huge", "giant", "giant-384"
        num_frames: Number of frames to sample. Default: 64
        alpha1: Weight for first moment. Default: 1.0
        alpha2: Weight for second moment. Default: 8.0
        alpha3: Weight for third moment. Default: 4.0
    """

    MODEL_MAP = {
        "large": "facebook/vjepa2-vitl-fpc64-256",
        "huge": "facebook/vjepa2-vith-fpc64-256",
        "giant": "facebook/vjepa2-vitg-fpc64-256",
        "giant-384": "facebook/vjepa2-vitg-fpc64-384",
    }

    HIDDEN_DIM_MAP = {
        "large": 1024,
        "huge": 1280,
        "giant": 1408,
        "giant-384": 1408,
    }

    def __init__(
        self,
        model_size: str = "large",
        num_frames: int = 64,
        alpha1: float = 1.0,
        alpha2: float = 8.0,
        alpha3: float = 4.0,
        aggregation: str = "concat",
    ):
        super().__init__(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, aggregation=aggregation)
        self.model_size = model_size
        self.num_frames = num_frames

        # Get model config
        model_name = self.MODEL_MAP.get(model_size, self.MODEL_MAP["large"])
        self.hidden_dim = self.HIDDEN_DIM_MAP.get(model_size, 1024)

        # Spatial patches: 16x16 = 256 for 256px input
        self.spatial_patches = 256

        # Load V-JEPA2 model and processor
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.processor = AutoVideoProcessor.from_pretrained(model_name)

        self.model.to(self.device).eval()

    def embed_video(self, video_frames: list) -> torch.Tensor:
        """Embed video using V-JEPA2 patch features.

        Args:
            video_frames: List of PIL Images.

        Returns:
            Video embedding tensor.
        """
        # Sample frames uniformly
        video_frames = sample_frames_uniformly(video_frames, num_frames=self.num_frames)

        # Process frames
        inputs = self.processor(video_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, skip_predictor=True)
            hidden_states = outputs.last_hidden_state.float().squeeze(0)  # (T*P, D)

            # Compute temporal dimension dynamically
            num_patches = hidden_states.shape[0]
            temporal_patches = num_patches // self.spatial_patches

            # Reshape to (T, P, D)
            patch_tokens = hidden_states.view(
                temporal_patches,
                self.spatial_patches,
                self.hidden_dim
            )

        # Compute moments
        return self.compute_moments(patch_tokens)
