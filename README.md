# AsymLightGCN Reproducibility Package

This repository contains a compact public package for reproducing AsymLightGCN experiments under sparse user-item interaction data.

It includes the model implementation, RecBole configuration files, benchmark and evaluation scripts, aggregate result tables, and data-access notes. It intentionally does not include raw datasets, prepared interaction files, semantic embeddings, or model checkpoints.

## What Is Included

- `asym_model/` - AsymLightGCN implementation and related model utilities.
- root `*.py` scripts - benchmark, evaluation, statistical analysis, and result-preparation entry points.
- `scripts/` - auxiliary/demo scripts copied from the working project.
- `configs/` - base RecBole configuration and Python dependencies.
- `build_semantic_embeddings.py` and `build_semantic_centroids.py` - generic scripts for rebuilding semantic item artifacts from a prepared `item_id,text` CSV.
- `results/` - aggregate evaluation tables and statistical summaries.
- `docs/` - data-access and artifact-policy notes.
- `tests/` - lightweight configuration tests.

## What Is Not Included

The following files are excluded on purpose:

- raw Amazon, Movies, and Yelp data;
- prepared RecBole `.inter`, `.item`, `.user` files;
- semantic embeddings and centroid tensors (`.pt`, `.npy`, `.pkl`);
- trained checkpoints (`.pth`, `.pt`);
- local logs, caches, virtual environments, and credentials.

See `docs/DATA_ACCESS.md` for instructions and rationale.

## Repository URL

This package is published as a separate public repository:

```text
https://github.com/nIBOP/asym-lightgcn-reproducibility
```

For a persistent archival identifier, create a tagged GitHub release and archive it through Zenodo.

## Environment

Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments use PyTorch, RecBole, sentence-transformers, pandas, NumPy, scikit-learn, SciPy, and Matplotlib.

## Expected Data Layout

After downloading and preparing the source datasets, place artifacts in the same paths used by the scripts:

```text
dataset_prep/clean_movies/
clean_amazon/
clean_yelp/
reduced_datasets/cold_preserving/
saved/
train_logs/
```

Large artifacts in these directories should remain local and untracked.

## Main Reproduction Steps

1. Download the source datasets from their official pages listed in `docs/DATA_ACCESS.md`.
2. Rebuild cleaned RecBole files and semantic embeddings following the project scripts and configuration paths.
3. Validate the prepared setup:

Example semantic artifact commands after preparing `item_texts.csv`:

```powershell
python build_semantic_embeddings.py --input-csv item_texts.csv --output-pt reduced_datasets/cold_preserving/clean_amazon_reduced_min5/semantic_embeddings.pt --normalize
python build_semantic_centroids.py --embeddings-pt reduced_datasets/cold_preserving/clean_amazon_reduced_min5/semantic_embeddings.pt --centroids-pt reduced_datasets/cold_preserving/clean_amazon_reduced_min5/cluster_centroids.pt --item-mapping reduced_datasets/cold_preserving/clean_amazon_reduced_min5/clean_amazon_reduced_min5.item --n-clusters 160 --seed 42
```

```powershell
python validate_final_full_graph_setup.py --datasets amazon_reduced_min5 movies_reduced yelp_reduced
```

4. Run the reduced benchmark protocol:

```powershell
python run_final_full_graph_benchmarks.py --datasets amazon_reduced_min5 movies_reduced yelp_reduced --models BPR LightGCN NCL AsymLightGCN --epochs 90 --train-batch-size 8192 --eval-batch-size 8192 --eval-step 15 --stopping-step 2
```

5. Recompute diagnostic controls and aggregate statistics:

```powershell
python evaluate_mostpop_independent.py
python analyze_seed_significance.py
python evaluate_user_level_significance.py
python prepare_amazon_threshold_assets.py
```

The exact command set may need adjustment depending on which artifacts are restored locally. Aggregate reference outputs are included in `results/`.

## Citation

Use `CITATION.cff` for software-package citation.

## License

Code is released under the MIT License. Dataset licenses and terms remain governed by the original dataset providers.
