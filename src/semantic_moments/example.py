"""Example usage of SemanticMoments."""

import argparse
import torch

from semantic_moments import (
    DINOEmbedder,
    VideoMAEEmbedder,
    VJEPA2Embedder,
    load_video_frames,
    embed_videos,
    compute_similarity_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="SemanticMoments example")
    parser.add_argument("videos", nargs="+", help="Video file paths")
    parser.add_argument(
        "--backbone",
        choices=["dino", "videomae", "vjepa2"],
        default="dino",
        help="Backbone to use",
    )
    parser.add_argument(
        "--aggregation",
        choices=["concat", "sum"],
        default="concat",
        help="Moment aggregation method",
    )
    args = parser.parse_args()

    # Initialize embedder
    print(f"Initializing {args.backbone} embedder...")
    if args.backbone == "dino":
        embedder = DINOEmbedder(aggregation=args.aggregation)
    elif args.backbone == "videomae":
        embedder = VideoMAEEmbedder(aggregation=args.aggregation)
    else:
        embedder = VJEPA2Embedder(aggregation=args.aggregation)

    # Embed videos
    print(f"Embedding {len(args.videos)} videos...")
    embeddings = embed_videos(embedder, args.videos)
    print(f"Embedding shape: {embeddings.shape}")

    # Compute similarity matrix
    if len(args.videos) > 1:
        similarity = compute_similarity_matrix(embeddings)
        print("\nSimilarity matrix:")
        print(similarity.cpu().numpy())

        # Show most similar pairs
        n = len(args.videos)
        for i in range(n):
            for j in range(i + 1, n):
                print(f"  {args.videos[i]} <-> {args.videos[j]}: {similarity[i, j]:.4f}")


if __name__ == "__main__":
    main()
