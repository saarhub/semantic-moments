# SemanticMoments

> **Saar Huberman, Kfir Goldberg, Or Patashnik, Sagie Benaim, Ron Mokady**

<a href="https://arxiv.org/abs/2602.09146"><img src="https://img.shields.io/badge/arXiv-2602.09146-b31b1b.svg"></a>

<p align="center">
  <img src="teaser.gif" width="600">
</p>

> Retrieving videos based on semantic motion is a fundamental, yet unsolved, problem. Existing video representation approaches overly rely on static appearance and scene context rather than motion dynamics, a bias inherited from their training data and objectives. Conversely, traditional motion-centric inputs like optical flow lack the semantic grounding needed to understand high-level motion. To demonstrate this inherent bias, we introduce the **SimMotion** benchmarks, combining controlled synthetic data with a new human-annotated real-world dataset. We show that existing models perform poorly on these benchmarks, often failing to disentangle motion from appearance. To address this gap, we propose **SemanticMoments**, a simple, training-free method that computes temporal statistics (specifically, higher-order moments) over features from pre-trained semantic models. Across our benchmarks, SemanticMoments consistently outperforms existing RGB, flow, and text-supervised methods. This demonstrates that temporal statistics in a semantic feature space provide a scalable and perceptually grounded foundation for motion-centric video understanding.

## Method

1. Extract patch features from a pretrained backbone (DINOv2, VideoMAE, V-JEPA2)
2. Compute temporal moments per patch: mean (M1), std (M2), skewness (M3)
3. Spatially aggregate (mean over patches)
4. Normalize and weight: α1=1, α2=8, α3=4
5. Concatenate and L2 normalize

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv), a modern Python package manager.

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/semantic-moments.git
cd semantic-moments

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync
```

This will install all required packages and lock their exact versions using `uv.lock`.

**Alternative (pip):**
```bash
pip install -e .
```

## Quick Start

**CLI:**
```bash
uv run python -m semantic_moments.example video1.mp4 video2.mp4 --backbone dino
```

**Python:**
```python
from semantic_moments import DINOEmbedder, load_video_frames, embed_videos, compute_similarity_matrix

# Initialize embedder
embedder = DINOEmbedder()

# Load and embed a single video
frames = load_video_frames("video.mp4", num_frames=32)
embedding = embedder.embed_video(frames)
print(f"Embedding shape: {embedding.shape}")  # (3072,) for concat, (1024,) for sum

# Embed multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
embeddings = embed_videos(embedder, video_paths)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 3072)

# Compute similarity matrix
similarity = compute_similarity_matrix(embeddings)
print(similarity)
```

## Available Backbones

| Backbone | Class | Default Frames | Feature Dim | Embedding Dim |
|----------|-------|----------------|-------------|---------------|
| DINOv2 | `DINOEmbedder` | 32 | 1024 | 3072 (concat) / 1024 (sum) |
| VideoMAE | `VideoMAEEmbedder` | 16 | 768 | 2304 (concat) / 768 (sum) |
| V-JEPA2 | `VJEPA2Embedder` | 64 | 1024 | 3072 (concat) / 1024 (sum) |

## Configuration

### Moment Weights

```python
# Default weights (paper): α1=1, α2=8, α3=4
embedder = DINOEmbedder(alpha1=1.0, alpha2=8.0, alpha3=4.0)

# Custom weights
embedder = DINOEmbedder(alpha1=1.0, alpha2=1.0, alpha3=1.0)
```

### Aggregation Mode

```python
# Concatenate moments (default) - output dim = 3 * feature_dim
embedder = DINOEmbedder(aggregation="concat")

# Sum moments - output dim = feature_dim
embedder = DINOEmbedder(aggregation="sum")
```

### Model Variants

```python
# DINOv2 variants
embedder = DINOEmbedder(model_name="dinov2_vitl14_reg")  # default
embedder = DINOEmbedder(model_name="dinov2_vitg14_reg")  # larger

# V-JEPA2 variants
embedder = VJEPA2Embedder(model_size="large")   # default
embedder = VJEPA2Embedder(model_size="huge")
embedder = VJEPA2Embedder(model_size="giant")
```

## API Reference

### Embedders

All embedders share the same interface:

```python
class Embedder:
    def embed_video(self, video_frames: list) -> torch.Tensor:
        """Embed a video from a list of PIL Images."""
        ...
```

### Utilities

```python
def load_video_frames(
    video_path: str,
    num_frames: int = 32,
    target_size: tuple = (224, 224),
    return_pil: bool = True,
) -> list:
    """Load video frames uniformly sampled from a video file."""

def embed_videos(
    embedder,
    video_paths: list,
    num_frames: int = None,
    target_size: tuple = (224, 224),
    show_progress: bool = True,
) -> torch.Tensor:
    """Embed multiple videos. Returns tensor of shape (N, D)."""

def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix. Input should be L2-normalized."""
```

## SimMotion Benchmark

We introduce **SimMotion**, a benchmark for evaluating motion similarity in videos.

Download from HuggingFace:
```bash
huggingface-cli download Shuberman/SimMotion-Synthetic --repo-type dataset --local-dir SimMotion_Synthetic_benchmark
huggingface-cli download Shuberman/SimMotion-Real --repo-type dataset --local-dir SimMotion_Real_benchmark
```

| Dataset | Triplets | Categories |
|---------|----------|------------|
| SimMotion-Synthetic | 250 | static_object, dynamic_attribute, dynamic_object, view, scene_style |
| SimMotion-Real | 40 | real-world videos |

### Evaluate Your Method

Benchmark your own video embedder on SimMotion:

```python
from semantic_moments import SimMotionSynthetic, SimMotionReal

# Your embedder must have: embed_video(frames: list[PIL.Image]) -> torch.Tensor
class MyEmbedder:
    def embed_video(self, frames):
        # Your embedding logic
        return embedding  # 1D tensor

# Evaluate on SimMotion-Synthetic
dataset = SimMotionSynthetic("SimMotion_Synthetic_benchmark")
results = dataset.evaluate(my_embedder, num_frames=32, save_to="results/my_method.json")
# Prints per-category Recall@1 and saves to JSON

# Evaluate on SimMotion-Real (with optional distractors)
dataset = SimMotionReal("SimMotion_Real_benchmark")
results = dataset.evaluate(my_embedder, num_frames=32, save_to="results/my_method_real.json")
```

Results are saved as JSON:
```json
{
  "dataset": "SimMotion-Synthetic",
  "embedder": "MyEmbedder",
  "num_frames": 32,
  "timestamp": "2026-03-02T...",
  "results": {
    "static_object": 85.0,
    "dynamic_attribute": 90.0,
    ...
    "average": 84.4
  }
}
```

## Citation

```bibtex
@article{huberman2026semanticmoments,
  title={SemanticMoments: Training-Free Motion Similarity via Third Moment Features},
  author={Huberman, Saar and Goldberg, Kfir and Patashnik, Or and Benaim, Sagie and Mokady, Ron},
  journal={arXiv preprint arXiv:2602.09146},
  year={2026}
}
```

## License

MIT
