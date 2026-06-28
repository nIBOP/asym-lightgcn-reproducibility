# Artifact Policy

## Include in the Public Repository

- Python source code required to define and evaluate AsymLightGCN.
- Configuration files and dataset path specifications.
- Scripts used to prepare article tables and statistical summaries.
- Aggregate result tables used in the article.
- Figures used in the article.
- Reproducibility manifest and response-to-reviewer material.

## Exclude from the Public Repository

- Raw datasets downloaded from Kaggle or Yelp.
- RecBole prepared interaction files.
- Semantic embeddings and cluster-centroid tensors.
- Model checkpoints.
- TensorBoard logs.
- Personal credentials and API keys.
- Temporary notebooks, scratch files, local archives, and obsolete drafts.

## Optional External Archive

If a journal or reviewer requires heavier artifacts, create a separate Zenodo archive with restricted or clearly licensed contents. Do not mix large or license-sensitive artifacts into the GitHub code repository.

Recommended split:

- GitHub: code, configs, scripts, aggregate results, documentation.
- Zenodo: tagged snapshot of GitHub and, only if legally allowed, additional deterministic split metadata.
