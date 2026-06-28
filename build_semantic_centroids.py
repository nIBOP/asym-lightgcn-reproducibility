"""Build semantic centroids and RecBole item-cluster mapping.

Input format:
    torch-saved dict from build_semantic_embeddings.py:
    {str(item_id): numpy.ndarray}

Outputs:
    - cluster_centroids.pt: torch.FloatTensor[n_clusters, dim]
    - item mapping TSV with columns `item_id:token` and `cluster_id:token`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-pt", required=True, type=Path)
    parser.add_argument("--centroids-pt", required=True, type=Path)
    parser.add_argument("--item-mapping", required=True, type=Path)
    parser.add_argument("--n-clusters", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = torch.load(args.embeddings_pt, map_location="cpu")
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError("Expected a non-empty dict of item_id -> embedding")

    item_ids = list(map(str, embeddings.keys()))
    matrix = np.vstack([np.asarray(embeddings[item_id], dtype=np.float32) for item_id in item_ids])

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init="auto",
    )
    labels = kmeans.fit_predict(matrix)

    args.centroids_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32), args.centroids_pt)

    args.item_mapping.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "item_id:token": item_ids,
            "cluster_id:token": labels.astype(int),
        }
    ).to_csv(args.item_mapping, sep="\t", index=False)

    print(f"Saved centroids to {args.centroids_pt}")
    print(f"Saved item mapping to {args.item_mapping}")


if __name__ == "__main__":
    main()
