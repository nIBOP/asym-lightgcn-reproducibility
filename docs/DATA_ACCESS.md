# Data Access and Redistribution Policy

This package does not redistribute raw datasets or prepared interaction files.

## Why Raw Data Are Excluded

The source datasets are publicly accessible from their providers, but public availability is not the same as permission to redistribute a copy in another repository. In particular, dataset providers may require users to accept their own terms or may restrict transfer of the raw files to third parties.

For this reason, the public reproducibility repository should contain:

- source links and access instructions;
- preparation scripts;
- filtering rules and configuration files;
- split protocol descriptions;
- aggregate statistics and selected evaluation results.

It should not contain:

- raw dataset archives or extracted raw records;
- prepared `.inter`, `.item`, `.user` files;
- semantic embedding tensors;
- trained checkpoints derived from restricted data;
- Kaggle credentials or API tokens.

## Source Datasets

### Amazon Books Reviews

Source page:
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

Recommended public-repo treatment:

- include source URL, access date, and filtering rules;
- do not include the downloaded Kaggle files;
- do not include generated RecBole `.inter`/`.item` files unless the dataset license is explicitly checked and permits redistribution.

### The Movies Dataset

Source page:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data

Recommended public-repo treatment:

- include source URL, access date, and preparation instructions;
- do not include the raw Kaggle archive;
- include only aggregate statistics and scripts.

### Yelp Open Dataset

Source page:
https://business.yelp.com/data/resources/open-dataset/

Recommended public-repo treatment:

- require users to download the dataset directly from Yelp;
- do not redistribute raw Yelp JSON files;
- do not redistribute prepared files that contain Yelp records;
- include only scripts, parameters, and aggregate statistics necessary for academic reproducibility.

## Semantic Embeddings and Centroids

Semantic embeddings and centroids are derived from source metadata and can be large. They should not be committed to Git. The package should instead record:

- the sentence-transformers model name: `sentence-transformers/all-MiniLM-L6-v2`;
- text fields used to build item descriptions;
- expected output paths;
- control statistics for the resulting slices.

The public package includes generic scripts for this stage:

- `build_semantic_embeddings.py`: converts a prepared `item_id,text` CSV into `semantic_embeddings.pt`;
- `build_semantic_centroids.py`: clusters embeddings and writes `cluster_centroids.pt` plus the RecBole item mapping.

## Split Files

If split assignment files are generated, prefer one of two approaches:

1. publish deterministic split-generation scripts with fixed seeds; or
2. publish minimal split identifiers only if the underlying dataset terms permit derived split redistribution.

For the current release, the safer default is approach 1: describe the split protocol and seeds, and include aggregate split statistics.
