"""Build item semantic embeddings from an item_id,text CSV file.

Input CSV format:
    item_id,text
    123,"title category description ..."

Output format:
    torch-saved dict: {str(item_id): numpy.ndarray}

This matches the format expected by AsymLightGCN's `semantic_embs_path`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-pt", required=True, type=Path)
    parser.add_argument("--id-col", default="item_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    missing = {args.id_col, args.text_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    item_ids = frame[args.id_col].astype(str).tolist()
    texts = frame[args.text_col].fillna("").astype(str).tolist()

    model = SentenceTransformer(args.model)
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    )

    output = {item_id: vector for item_id, vector in zip(item_ids, vectors)}
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_pt)
    print(f"Saved {len(output)} item embeddings to {args.output_pt}")


if __name__ == "__main__":
    main()
