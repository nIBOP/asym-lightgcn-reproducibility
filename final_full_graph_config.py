from copy import deepcopy


DATASET_SPECS = {
    "movies": {
        "key": "movies",
        "dataset_name": "clean_movies",
        "display_name": "Clean Movies",
        "data_path": "dataset_prep",
        "dataset_dir": "dataset_prep/clean_movies",
        "centroids_path": "dataset_prep/cluster_centroids.pt",
        "item_mapping_path": "dataset_prep/clean_movies/clean_movies.item",
        "semantic_embs_path": "dataset_prep/clean_movies/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_full_graph/clean_movies",
    },
    "amazon": {
        "key": "amazon",
        "dataset_name": "clean_amazon",
        "display_name": "Clean Amazon",
        "data_path": ".",
        "dataset_dir": "clean_amazon",
        "centroids_path": "clean_amazon/cluster_centroids_amazon.pt",
        "item_mapping_path": "clean_amazon/clean_amazon.item",
        "semantic_embs_path": "clean_amazon/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_full_graph/clean_amazon",
    },
    "yelp": {
        "key": "yelp",
        "dataset_name": "clean_yelp",
        "display_name": "Clean Yelp",
        "data_path": ".",
        "dataset_dir": "clean_yelp",
        "centroids_path": "clean_yelp/cluster_centroids_yelp.pt",
        "item_mapping_path": "clean_yelp/clean_yelp.item",
        "semantic_embs_path": "clean_yelp/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_full_graph/clean_yelp",
    },
    "movies_reduced": {
        "key": "movies_reduced",
        "dataset_name": "clean_movies_reduced",
        "display_name": "Reduced Movies",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_movies_reduced",
        "centroids_path": "reduced_datasets/cold_preserving/clean_movies_reduced/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_movies_reduced/clean_movies_reduced.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_movies_reduced/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_movies_reduced",
    },
    "amazon_reduced": {
        "key": "amazon_reduced",
        "dataset_name": "clean_amazon_reduced",
        "display_name": "Reduced Amazon",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced/clean_amazon_reduced.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced",
    },
    "amazon_reduced_min3": {
        "key": "amazon_reduced_min3",
        "dataset_name": "clean_amazon_reduced_min3",
        "display_name": "Reduced Amazon (min user interactions >= 3)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min3",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min3/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min3/clean_amazon_reduced_min3.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min3/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min3",
    },
    "amazon_reduced_min4": {
        "key": "amazon_reduced_min4",
        "dataset_name": "clean_amazon_reduced_min4",
        "display_name": "Reduced Amazon (min user interactions >= 4)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min4",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min4/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min4/clean_amazon_reduced_min4.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min4/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min4",
    },
    "amazon_reduced_min5": {
        "key": "amazon_reduced_min5",
        "dataset_name": "clean_amazon_reduced_min5",
        "display_name": "Reduced Amazon (min user interactions >= 5)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min5",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min5/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min5/clean_amazon_reduced_min5.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min5/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min5",
    },
    "amazon_reduced_min6": {
        "key": "amazon_reduced_min6",
        "dataset_name": "clean_amazon_reduced_min6",
        "display_name": "Reduced Amazon (min user interactions >= 6)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min6",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min6/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min6/clean_amazon_reduced_min6.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min6/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min6",
    },
    "amazon_reduced_min8": {
        "key": "amazon_reduced_min8",
        "dataset_name": "clean_amazon_reduced_min8",
        "display_name": "Reduced Amazon (min user interactions >= 8)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min8",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min8/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min8/clean_amazon_reduced_min8.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min8/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min8",
    },
    "amazon_reduced_min10": {
        "key": "amazon_reduced_min10",
        "dataset_name": "clean_amazon_reduced_min10",
        "display_name": "Reduced Amazon (min user interactions >= 10)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_amazon_reduced_min10",
        "centroids_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min10/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min10/clean_amazon_reduced_min10.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_amazon_reduced_min10/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_amazon_reduced_min10",
    },
    "yelp_reduced": {
        "key": "yelp_reduced",
        "dataset_name": "clean_yelp_reduced",
        "display_name": "Reduced Yelp",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_yelp_reduced",
        "centroids_path": "reduced_datasets/cold_preserving/clean_yelp_reduced/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_yelp_reduced/clean_yelp_reduced.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_yelp_reduced/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_yelp_reduced",
    },
    "yelp_reduced_min3": {
        "key": "yelp_reduced_min3",
        "dataset_name": "clean_yelp_reduced_min3",
        "display_name": "Reduced Yelp (min user interactions >= 3)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_yelp_reduced_min3",
        "centroids_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min3/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min3/clean_yelp_reduced_min3.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min3/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_yelp_reduced_min3",
    },
    "yelp_reduced_min5": {
        "key": "yelp_reduced_min5",
        "dataset_name": "clean_yelp_reduced_min5",
        "display_name": "Reduced Yelp (min user interactions >= 5)",
        "data_path": "reduced_datasets/cold_preserving",
        "dataset_dir": "reduced_datasets/cold_preserving/clean_yelp_reduced_min5",
        "centroids_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min5/cluster_centroids.pt",
        "item_mapping_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min5/clean_yelp_reduced_min5.item",
        "semantic_embs_path": "reduced_datasets/cold_preserving/clean_yelp_reduced_min5/semantic_embeddings.pt",
        "checkpoint_dir": "saved/final_reduced/clean_yelp_reduced_min5",
    },
}


DEFAULT_DATASET_ORDER = ["movies", "amazon", "yelp"]
DEFAULT_REDUCED_DATASET_ORDER = ["movies_reduced", "amazon_reduced", "yelp_reduced"]
ALL_DATASET_KEYS = list(DATASET_SPECS.keys())
DEFAULT_NCL_TRAIN_BATCH_SIZE = 1920
DEFAULT_NCL_CONTRASTIVE_MIB_LIMIT = 2048.0
DEFAULT_NCL_KMEANS_DEVICE = "auto"
FAISS_RECOMMENDED_MIN_POINTS_PER_CENTROID = 39
FLOAT32_BYTES = 4


def recommended_ncl_max_clusters(
    user_num,
    item_num,
    min_points_per_centroid=FAISS_RECOMMENDED_MIN_POINTS_PER_CENTROID,
):
    user_points = max(int(user_num) - 1, 1)
    item_points = max(int(item_num) - 1, 1)
    min_points_per_centroid = int(min_points_per_centroid)
    if min_points_per_centroid <= 0:
        raise ValueError("min_points_per_centroid must be positive")
    return max(1, min(user_points, item_points) // min_points_per_centroid)


def choose_ncl_num_clusters(user_num, item_num, divisor=50, cap=500, floor=50):
    user_points = max(int(user_num) - 1, 1)
    item_points = max(int(item_num) - 1, 1)
    safe_max = max(floor, min(user_points, item_points) // divisor)
    return max(floor, min(cap, safe_max))


def choose_ncl_train_batch_size(train_batch_size, ncl_train_batch_size=DEFAULT_NCL_TRAIN_BATCH_SIZE):
    train_batch_size = int(train_batch_size)
    ncl_train_batch_size = int(ncl_train_batch_size)
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if ncl_train_batch_size <= 0:
        raise ValueError("ncl_train_batch_size must be positive")
    return min(train_batch_size, ncl_train_batch_size)


def choose_ncl_kmeans_device(
    num_clusters,
    user_num,
    item_num,
    requested=DEFAULT_NCL_KMEANS_DEVICE,
    min_points_per_centroid=FAISS_RECOMMENDED_MIN_POINTS_PER_CENTROID,
):
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("requested must be one of: auto, cpu, gpu")
    if requested != "auto":
        return requested

    num_clusters = int(num_clusters)
    user_points = max(int(user_num) - 1, 1)
    item_points = max(int(item_num) - 1, 1)
    min_points_per_centroid = int(min_points_per_centroid)
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if min_points_per_centroid <= 0:
        raise ValueError("min_points_per_centroid must be positive")

    if num_clusters > recommended_ncl_max_clusters(
        user_num=user_num,
        item_num=item_num,
        min_points_per_centroid=min_points_per_centroid,
    ):
        return "cpu"
    return "gpu"


def estimate_ncl_contrastive_matrix_mib(batch_size, user_num, item_num):
    batch_size = int(batch_size)
    user_num = int(user_num)
    item_num = int(item_num)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if user_num <= 0 or item_num <= 0:
        raise ValueError("user_num and item_num must be positive")

    largest_candidate_axis = max(user_num, item_num)
    return batch_size * largest_candidate_axis * FLOAT32_BYTES / (1024**2)


def build_base_config(spec, *, seed, epochs, train_batch_size, eval_batch_size, eval_step, stopping_step):
    return {
        "data_path": spec["data_path"],
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "centroids_path": spec["centroids_path"],
        "item_mapping_path": spec["item_mapping_path"],
        "semantic_embs_path": spec["semantic_embs_path"],
        "checkpoint_dir": spec["checkpoint_dir"],
        "seed": seed,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_step": eval_step,
        "stopping_step": stopping_step,
        "batch_log_interval": 0,
        "log_validation_metrics": False,
        "log_epoch_summary": False,
    }


def resolve_dataset_specs(dataset_keys):
    return [deepcopy(DATASET_SPECS[key]) for key in dataset_keys]
