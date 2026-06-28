import unittest

from final_full_graph_config import (
    DATASET_SPECS,
    DEFAULT_DATASET_ORDER,
    DEFAULT_NCL_TRAIN_BATCH_SIZE,
    build_base_config,
    choose_ncl_kmeans_device,
    choose_ncl_train_batch_size,
    choose_ncl_num_clusters,
    estimate_ncl_contrastive_matrix_mib,
    recommended_ncl_max_clusters,
    resolve_dataset_specs,
)


class FinalFullGraphConfigTests(unittest.TestCase):
    def test_default_dataset_order_matches_specs(self):
        self.assertEqual(DEFAULT_DATASET_ORDER, ["movies", "amazon", "yelp"])
        for key in DEFAULT_DATASET_ORDER:
            self.assertIn(key, DATASET_SPECS)

    def test_choose_ncl_num_clusters_respects_floor(self):
        self.assertEqual(choose_ncl_num_clusters(user_num=10, item_num=10), 50)

    def test_choose_ncl_num_clusters_respects_cap(self):
        self.assertEqual(choose_ncl_num_clusters(user_num=1_000_000, item_num=1_000_000), 500)

    def test_choose_ncl_num_clusters_scales_with_dataset_size(self):
        self.assertEqual(choose_ncl_num_clusters(user_num=3928, item_num=2700), 53)
        self.assertEqual(choose_ncl_num_clusters(user_num=27335, item_num=10246), 204)

    def test_choose_ncl_train_batch_size_caps_main_batch(self):
        self.assertEqual(DEFAULT_NCL_TRAIN_BATCH_SIZE, 1920)
        self.assertEqual(choose_ncl_train_batch_size(8192), 1920)
        self.assertEqual(choose_ncl_train_batch_size(8192, 1024), 1024)
        self.assertEqual(choose_ncl_train_batch_size(512, 1024), 512)

    def test_choose_ncl_kmeans_device_uses_cpu_for_sparse_overclustering(self):
        self.assertEqual(
            choose_ncl_kmeans_device(
                num_clusters=1000,
                user_num=265880,
                item_num=6388,
            ),
            "cpu",
        )

    def test_recommended_ncl_max_clusters_uses_sparsest_side(self):
        self.assertEqual(
            recommended_ncl_max_clusters(user_num=265880, item_num=6388),
            163,
        )

    def test_choose_ncl_kmeans_device_uses_gpu_when_faiss_has_enough_points(self):
        self.assertEqual(
            choose_ncl_kmeans_device(
                num_clusters=127,
                user_num=265880,
                item_num=6388,
            ),
            "gpu",
        )

    def test_choose_ncl_kmeans_device_respects_explicit_request(self):
        self.assertEqual(choose_ncl_kmeans_device(1000, 265880, 6388, requested="gpu"), "gpu")

    def test_estimate_ncl_contrastive_matrix_mib_uses_largest_axis(self):
        estimated_mib = estimate_ncl_contrastive_matrix_mib(
            batch_size=1024,
            user_num=265880,
            item_num=6388,
        )
        self.assertAlmostEqual(estimated_mib, 1038.59375)

    def test_build_base_config_embeds_runtime_controls(self):
        spec = DATASET_SPECS["amazon"]
        config = build_base_config(
            spec,
            seed=42,
            epochs=120,
            train_batch_size=4096,
            eval_batch_size=4096,
            eval_step=10,
            stopping_step=4,
        )
        self.assertEqual(config["data_path"], ".")
        self.assertEqual(config["epochs"], 120)
        self.assertEqual(config["eval_step"], 10)
        self.assertEqual(config["stopping_step"], 4)
        self.assertEqual(config["checkpoint_dir"], spec["checkpoint_dir"])

    def test_resolve_dataset_specs_returns_independent_copies(self):
        specs = resolve_dataset_specs(["movies", "amazon"])
        self.assertEqual([spec["key"] for spec in specs], ["movies", "amazon"])
        specs[0]["display_name"] = "Changed"
        self.assertNotEqual(specs[0]["display_name"], DATASET_SPECS["movies"]["display_name"])


if __name__ == "__main__":
    unittest.main()
