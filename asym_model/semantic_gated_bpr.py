import pandas as pd
import torch
import torch.nn as nn
from recbole.model.general_recommender import BPR


def _cfg(config, key, default):
    try:
        value = config[key]
    except KeyError:
        return default
    return default if value is None else value


class SemanticGatedBPR(BPR):
    """BPR-MF with degree-aware semantic item gating and no graph propagation."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.semantic_embs_path = _cfg(config, "semantic_embs_path", None)
        self.item_mapping_path = _cfg(config, "item_mapping_path", None)
        self.use_semantic_gating = bool(_cfg(config, "use_semantic_gating", True))
        self.gating_w_max = float(_cfg(config, "gating_w_max", 0.8))

        self.raw_semantic_embs = None
        self.semantic_gating_proj = None
        if self.use_semantic_gating and self.semantic_embs_path:
            self._load_raw_semantics(dataset)
        self._init_degree_aware_gating(dataset)

    def _load_raw_semantics(self, dataset):
        try:
            raw_embs_dict = torch.load(self.semantic_embs_path, map_location="cpu")
            example_emb = next(iter(raw_embs_dict.values()))
            raw_dim = int(len(example_emb))

            temp_embs = torch.zeros(self.n_items, raw_dim, device=self.device)
            token2id = dataset.field2token_id[dataset.iid_field]
            mapped_count = 0
            for token, iid in token2id.items():
                if token == "[PAD]":
                    continue
                semantic_vector = raw_embs_dict.get(str(token))
                if semantic_vector is None:
                    continue
                temp_embs[iid] = torch.as_tensor(semantic_vector, dtype=torch.float32, device=self.device)
                mapped_count += 1

            if hasattr(self, "raw_semantic_embs"):
                delattr(self, "raw_semantic_embs")
            self.register_buffer("raw_semantic_embs", temp_embs, persistent=False)
            self.semantic_gating_proj = nn.Sequential(
                nn.Linear(raw_dim, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.LeakyReLU(0.2),
            ).to(self.device)
            print(f"[SemanticGatedBPR] mapped semantic vectors: {mapped_count}/{self.n_items}")
        except Exception as exc:
            print(f"[SemanticGatedBPR] semantic loading failed, falling back to BPR item embeddings: {exc}")
            self.use_semantic_gating = False
            self.raw_semantic_embs = None
            self.semantic_gating_proj = None

    def _init_degree_aware_gating(self, dataset):
        item_ids = dataset.inter_feat[dataset.iid_field]
        item_counts = torch.bincount(item_ids, minlength=self.n_items).float().to(self.device)
        self.register_buffer("item_counts", item_counts, persistent=False)

        if self.use_semantic_gating and self.raw_semantic_embs is not None:
            active_items = item_counts[item_counts > 0]
            gating_tau = torch.median(active_items).item() if len(active_items) > 0 else 1.0
            gating_weights = self.gating_w_max * torch.exp(-item_counts / (gating_tau + 1e-5))
            gating_weights = gating_weights.unsqueeze(1)
        else:
            gating_weights = torch.zeros(self.n_items, 1, device=self.device)
        self.register_buffer("gating_weights", gating_weights, persistent=False)

    def _all_item_embeddings(self):
        item_cf = self.item_embedding.weight
        if not self.use_semantic_gating or self.raw_semantic_embs is None or self.semantic_gating_proj is None:
            return item_cf
        item_sem = self.semantic_gating_proj(self.raw_semantic_embs)
        return (1.0 - self.gating_weights) * item_cf + self.gating_weights * item_sem

    def get_item_embedding(self, item):
        return self._all_item_embeddings()[item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        score = torch.matmul(user_e, self._all_item_embeddings().transpose(0, 1))
        return score.view(-1)
