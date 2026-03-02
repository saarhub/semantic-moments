"""Dataset loaders for SimMotion benchmark."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union, Dict

import torch
from tqdm import tqdm

from .utils import load_video_frames


def _save_results(results: Dict, save_to: str, embedder, num_frames: int, dataset_name: str):
    """Save evaluation results to JSON file."""
    save_path = Path(save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": dataset_name,
        "embedder": embedder.__class__.__name__,
        "num_frames": num_frames,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {save_path}")


HF_REPO_SYNTHETIC = "Shuberman/SimMotion-Synthetic"
HF_REPO_REAL = "Shuberman/SimMotion-Real"


def _embed_videos(embedder, paths: List[str], num_frames: int) -> torch.Tensor:
    """Embed videos and return stacked normalized tensor."""
    embeddings = []
    for path in tqdm(paths, desc="Embedding videos"):
        frames = load_video_frames(path, num_frames=num_frames)
        emb = embedder.embed_video(frames)
        embeddings.append(emb)
    E = torch.stack(embeddings).to(torch.float32)
    return torch.nn.functional.normalize(E, dim=1)


def download_simmotion(
    dataset: str = "both",
    local_dir: str = ".",
) -> None:
    """Download SimMotion benchmark from HuggingFace.

    Args:
        dataset: Which dataset to download: "synthetic", "real", or "both"
        local_dir: Directory to download to
    """
    local_path = Path(local_dir)

    if dataset in ("synthetic", "both"):
        target = local_path / "SimMotion_Synthetic_benchmark"
        if not target.exists():
            print(f"Downloading SimMotion-Synthetic to {target}...")
            subprocess.run([
                "huggingface-cli", "download",
                HF_REPO_SYNTHETIC,
                "--repo-type", "dataset",
                "--local-dir", str(target),
            ], check=True)
        else:
            print(f"SimMotion-Synthetic already exists at {target}")

    if dataset in ("real", "both"):
        target = local_path / "SimMotion_Real_benchmark"
        if not target.exists():
            print(f"Downloading SimMotion-Real to {target}...")
            subprocess.run([
                "huggingface-cli", "download",
                HF_REPO_REAL,
                "--repo-type", "dataset",
                "--local-dir", str(target),
            ], check=True)
        else:
            print(f"SimMotion-Real already exists at {target}")


@dataclass
class Triplet:
    """A triplet of videos for motion similarity evaluation."""
    ref_path: str
    positive_path: str
    negative_path: str
    example_id: str
    category: Optional[str] = None


class SimMotionSynthetic:
    """SimMotion-Synthetic dataset loader.

    250 triplets (750 videos) across 5 categories:
    - static_object
    - dynamic_attribute
    - dynamic_object
    - view
    - scene_style

    Args:
        root: Path to SimMotion_Synthetic_benchmark directory.
        categories: List of categories to load. If None, loads all.
    """

    ALL_CATEGORIES = [
        "static_object",
        "dynamic_attribute",
        "dynamic_object",
        "view",
        "scene_style",
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.categories = categories or self.ALL_CATEGORIES
        self.triplets = self._load_triplets()

    def _load_triplets(self) -> List[Triplet]:
        triplets = []

        for category in self.categories:
            category_path = self.root / category
            if not category_path.exists():
                continue

            examples = sorted(
                [d for d in category_path.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0
            )

            for example_dir in examples:
                ref = example_dir / "ref.mp4"
                pos = example_dir / "positive.mp4"
                neg = example_dir / "negative.mp4"

                if ref.exists() and pos.exists() and neg.exists():
                    triplets.append(Triplet(
                        ref_path=str(ref),
                        positive_path=str(pos),
                        negative_path=str(neg),
                        example_id=example_dir.name,
                        category=category,
                    ))

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Triplet:
        return self.triplets[idx]

    def __iter__(self):
        return iter(self.triplets)

    def by_category(self, category: str) -> List[Triplet]:
        """Get triplets for a specific category."""
        return [t for t in self.triplets if t.category == category]

    def all_video_paths(self) -> List[str]:
        """Get all video paths."""
        paths = []
        for t in self.triplets:
            paths.extend([t.ref_path, t.positive_path, t.negative_path])
        return paths

    @torch.no_grad()
    def evaluate(
        self,
        embedder,
        num_frames: int = 32,
        save_to: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate Recall@1 on SimMotion-Synthetic.

        For each triplet, checks if positive is the nearest neighbor to ref
        among ALL videos in the dataset.

        Args:
            embedder: Embedder instance with embed_video() method.
            num_frames: Number of frames to sample per video.
            save_to: Optional path to save results as JSON.

        Returns:
            Dict mapping category -> Recall@1 (0-100), plus "average".
        """
        all_paths = self.all_video_paths()
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        # Embed all videos
        E = _embed_videos(embedder, all_paths, num_frames)

        # Similarity matrix
        S = E @ E.T

        # Evaluate per category
        results = {}
        for category in self.categories:
            triplets = self.by_category(category)
            correct = 0

            for t in triplets:
                ref_idx = path_to_idx[t.ref_path]
                pos_idx = path_to_idx[t.positive_path]

                sims = S[ref_idx].clone()
                sims[ref_idx] = -float("inf")  # mask self

                if sims.argmax().item() == pos_idx:
                    correct += 1

            acc = 100.0 * correct / len(triplets) if triplets else 0.0
            results[category] = acc
            print(f"{category}: {correct}/{len(triplets)} = {acc:.1f}%")

        avg = sum(results.values()) / len(results) if results else 0.0
        results["average"] = avg
        print(f"Average: {avg:.1f}%")

        if save_to:
            _save_results(results, save_to, embedder, num_frames, "SimMotion-Synthetic")

        return results


class SimMotionReal:
    """SimMotion-Real dataset loader.

    40 triplets of real videos.

    Args:
        root: Path to SimMotion_Real_benchmark directory.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.examples_dir = self.root / "examples"
        self.triplets = self._load_triplets()

    def _load_triplets(self) -> List[Triplet]:
        triplets = []

        if not self.examples_dir.exists():
            return triplets

        examples = sorted(
            [d for d in self.examples_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split("_")[-1]) if "_" in x.name else 0
        )

        for example_dir in examples:
            ref = example_dir / "ref.mp4"
            pos = example_dir / "positive.mp4"
            neg = example_dir / "negative.mp4"

            if ref.exists() and pos.exists() and neg.exists():
                triplets.append(Triplet(
                    ref_path=str(ref),
                    positive_path=str(pos),
                    negative_path=str(neg),
                    example_id=example_dir.name,
                    category="real",
                ))

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Triplet:
        return self.triplets[idx]

    def __iter__(self):
        return iter(self.triplets)

    def all_video_paths(self) -> List[str]:
        """Get all video paths."""
        paths = []
        for t in self.triplets:
            paths.extend([t.ref_path, t.positive_path, t.negative_path])
        return paths

    @torch.no_grad()
    def evaluate(
        self,
        embedder,
        num_frames: int = 32,
        distractors: Union[List[str], torch.Tensor, None] = None,
        save_to: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate Recall@1 on SimMotion-Real.

        For each triplet, checks if positive is the nearest neighbor to ref
        among all dataset videos plus optional distractors.

        Args:
            embedder: Embedder instance with embed_video() method.
            num_frames: Number of frames to sample per video.
            distractors: Optional external distractor videos. Either:
                - List of video paths (will be embedded)
                - Pre-computed embeddings tensor [N, D] (for efficiency)
                In the paper, we use Kinetics-400 test set as distractors.
            save_to: Optional path to save results as JSON.

        Returns:
            Dict with "real" -> Recall@1 (0-100).
        """
        all_paths = self.all_video_paths()
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        # Embed dataset videos
        E = _embed_videos(embedder, all_paths, num_frames)

        # Handle distractors
        if distractors is not None:
            if isinstance(distractors, torch.Tensor):
                E_dist = distractors.to(E.device).to(torch.float32)
                E_dist = torch.nn.functional.normalize(E_dist, dim=1)
            else:
                print(f"Embedding {len(distractors)} distractor videos...")
                E_dist = _embed_videos(embedder, distractors, num_frames)
            # Candidates = dataset videos + distractors
            C = torch.cat([E, E_dist], dim=0)
        else:
            C = E

        # Evaluate
        correct = 0
        for t in self.triplets:
            ref_idx = path_to_idx[t.ref_path]
            pos_idx = path_to_idx[t.positive_path]
            neg_idx = path_to_idx[t.negative_path]

            q = E[ref_idx].unsqueeze(0)
            sims = (q @ C.T).squeeze(0)

            # Mask ref, pos, neg in the dataset portion
            sims[ref_idx] = -float("inf")
            sims[pos_idx] = -float("inf")
            sims[neg_idx] = -float("inf")

            # Check if positive beats negative and all distractors
            pos_sim = (q @ E[pos_idx]).item()
            neg_sim = (q @ E[neg_idx]).item()
            max_other = sims.max().item()

            if pos_sim > neg_sim and pos_sim > max_other:
                correct += 1

        acc = 100.0 * correct / len(self.triplets) if self.triplets else 0.0
        print(f"SimMotion-Real: {correct}/{len(self.triplets)} = {acc:.1f}%")

        results = {"real": acc}

        if save_to:
            _save_results(results, save_to, embedder, num_frames, "SimMotion-Real")

        return results
