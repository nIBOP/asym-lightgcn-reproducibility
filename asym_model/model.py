import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from recbole.model.general_recommender.lightgcn import LightGCN


def _cfg(config, key, default):
    """Safely fetch optional config value with a default fallback."""
    try:
        value = config[key]
    except KeyError:
        return default
    return default if value is None else value

class AsymLightGCN(LightGCN):
    """
    Asymmetric LightGCN (AsymLightGCN)
    ----------------------------------
    Расширенная версия графовой модели LightGCN, которая решает проблемы популярности (Popularity Bias) 
    и холодного старта (Cold Start) с помощью:
    1. Semantic Gating: Подмешивание сырых текстовых эмбеддингов для редких фильмов.
    2. Adaptive Contrastive Loss: Семантическая SSL-регуляризация с помощью динамических кластеров.
    3. Homoscedastic Uncertainty: Автоматическая балансировка весов лоссов.
    4. Fractional Magnitude Calibration: Снижение влияния популярных фильмов на этапе предсказания.
    """
    def __init__(self, config, dataset):
        super(AsymLightGCN, self).__init__(config, dataset)
        
        # --- 1. Загрузка параметров из конфигурации ---
        self.centroids_path = config['centroids_path']
        self.item_mapping_path = _cfg(config, 'item_mapping_path', 'dataset_prep/clean_movies/clean_movies.item')
        
        # Параметры Semantic Contrastive Loss
        self.semantic_sampling = _cfg(config, 'semantic_sampling', 'uniform')
        self.ema_momentum = _cfg(config, 'ema_momentum', 0.99)
        self.use_centroid_ema = _cfg(config, 'use_centroid_ema', True)
        self.hit_quantile = _cfg(config, 'hit_quantile', 0.80)
        self.semantic_margin = _cfg(config, 'semantic_margin', 0.8)
        
        # Параметры весов семантического лосса (Alpha Strategy)
        self.alpha_strategy = _cfg(config, 'alpha_strategy', 'static')
        self.semantic_weight_cap = _cfg(config, 'semantic_weight_cap', 0.5)
        self.semantic_weight_floor = _cfg(config, 'semantic_weight_floor', 0.05)
        self.dynamic_cutoff_quantile = _cfg(config, 'dynamic_cutoff_quantile', 0.85)

        # --- 2. Инициализация механизма Semantic Gating ---
        self.semantic_embs_path = _cfg(config, 'semantic_embs_path', 'dataset_prep/clean_movies/semantic_embeddings.pt')
        self.use_semantic_gating = _cfg(config, 'use_semantic_gating', True)
        self.use_semantic_loss = _cfg(config, 'use_semantic_loss', True)
        self.gating_w_max = _cfg(config, 'gating_w_max', 0.8)

        # --- 3. Инициализация Fractional Magnitude Calibration ---
        self.magnitude_gamma = _cfg(config, 'magnitude_calibration_gamma', 0.25)

        # --- 4. Загрузка семантических кластеров (Centroids) ---
        self._load_semantic_centroids(dataset)

        # --- 5. Загрузка сырых семантических эмбеддингов ---
        self.raw_semantic_embs = None
        if self.use_semantic_gating:
            self._load_raw_semantics(dataset)

        # --- 6. Настройка Degree-Aware параметров (зависящих от популярности) ---
        self._init_degree_aware_params(config, dataset)

        # --- 7. Homoscedastic Uncertainty ---
        # Обучаемые параметры для автоматического подбора весов BPR и Semantic лоссов
        self.log_vars = nn.Parameter(torch.zeros(2))

    def _load_semantic_centroids(self, dataset):
        """Загрузка центроидов кластеров и маппинг фильмов к их кластерам."""
        try:
            self.semantic_centroids = torch.load(self.centroids_path).to(self.device)
            self.n_clusters = self.semantic_centroids.shape[0]
            centroid_dim = self.semantic_centroids.shape[1]

            # Проекция размерности, если центроиды не совпадают с размерностью эмбеддингов (Latent Dim)
            if centroid_dim != self.latent_dim:
                self.centroid_proj = nn.Sequential(
                    nn.Linear(centroid_dim, self.latent_dim * 2),
                    nn.LayerNorm(self.latent_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                ).to(self.device)
            else:
                self.centroid_proj = None
            print(f"[AsymLightGCN] Loaded {self.n_clusters} semantic clusters.")
        except Exception as e:
            print(f"[AsymLightGCN] Failed to load clusters: {e}")
            self.semantic_centroids = None

        if self.semantic_centroids is not None:
            self.item2cluster = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
            try:
                # Маппинг токенов (строковых ID) к внутренним ID (int) в RecBole
                df_map = pd.read_csv(self.item_mapping_path, sep='\t', dtype=str)
                token2id = dataset.field2token_id[dataset.iid_field]

                mapped_count = 0
                for _, row in df_map.iterrows():
                    token = row['item_id:token']
                    cluster_idx = int(row['cluster_id:token'])
                    if token in token2id:
                        iid = token2id[token]
                        self.item2cluster[iid] = cluster_idx
                        mapped_count += 1
                print(f"[AsymLightGCN] Mapped {mapped_count} items to clusters.")
            except Exception as e:
                print(f"[AsymLightGCN] Failed to map clusters: {e}")

            # --- Инициализация EMA буфера маяков (Dynamic Centroids) ---
            with torch.no_grad():
                if self.centroid_proj is not None:
                    init_dyn = self.centroid_proj(self.semantic_centroids)
                else:
                    init_dyn = self.semantic_centroids.clone()
            self.register_buffer('dynamic_centroids', init_dyn.clone())
            
    def _load_raw_semantics(self, dataset):
        """Загрузка сырых текстовых эмбеддингов для механизма гейтирования."""
        try:
            raw_embs_dict = torch.load(self.semantic_embs_path)
            example_emb = next(iter(raw_embs_dict.values()))
            raw_dim = len(example_emb)
            
            temp_embs = torch.zeros(self.n_items, raw_dim, device=self.device)
            token2id = dataset.field2token_id[dataset.iid_field]
            mapped_count = 0
            
            for token, iid in token2id.items():
                if token != '[PAD]' and str(token) in raw_embs_dict:
                    emb_tensor = torch.as_tensor(raw_embs_dict[str(token)], dtype=torch.float32, device=self.device)
                    temp_embs[iid] = emb_tensor
                    mapped_count += 1
            print(f"[AsymLightGCN] Mapped {mapped_count} raw semantic vectors.")
            
            if hasattr(self, 'raw_semantic_embs') and 'raw_semantic_embs' not in self._buffers:
                delattr(self, 'raw_semantic_embs')
            self.register_buffer('raw_semantic_embs', temp_embs, persistent=False)
            
            # Линейный слой проекции сырой текстовой размерности в графовую
            self.semantic_gating_proj = nn.Sequential(
                nn.Linear(raw_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.LeakyReLU(0.2)
            ).to(self.device)
        except Exception as e:
            print(f"[AsymLightGCN] Failed to load raw semantics: {e}")
            self.use_semantic_gating = False

    def _init_degree_aware_params(self, config, dataset):
        """Рассчет динамических параметров (Weights, Temparatures), зависящих от степени узла."""
        train_item_ids = dataset.inter_feat[dataset.iid_field]
        item_counts = torch.bincount(train_item_ids, minlength=self.n_items).float().to(self.device)
        self.item_counts = item_counts
        
        # Порог популярности для EMA-обновлений кластеров (хиты двигают кластеры)
        active_items = item_counts[item_counts > 0]
        self.hit_threshold = torch.quantile(active_items, self.hit_quantile).item() if len(active_items) > 0 else 0

        # -- Веса семантического лосса --
        if self.alpha_strategy == 'static':
            self.item_alpha_weights = torch.ones_like(item_counts)
        elif self.alpha_strategy == 'dynamic_leaky_log':
            if len(active_items) > 0:
                dynamic_cutoff = torch.quantile(active_items, self.dynamic_cutoff_quantile).item()
                log_counts = torch.log(item_counts + 1)
                log_cutoff = torch.log(torch.tensor(dynamic_cutoff + 1, device=self.device))
                raw_alphas = 1.0 - (log_counts / log_cutoff)
            else:
                raw_alphas = torch.ones_like(item_counts)
            self.item_alpha_weights = torch.clamp(raw_alphas, min=self.semantic_weight_floor, max=self.semantic_weight_cap)
        else:
            raw_alphas = 1.0 / torch.log(torch.e + item_counts)
            self.item_alpha_weights = torch.clamp(raw_alphas, max=self.semantic_weight_cap)

        # -- Индивидуальная Temperature (tau) для каждого фильма --
        self.tau_min = _cfg(config, 'tau_min', 0.2)
        self.tau_max = _cfg(config, 'tau_max', 0.8)
        
        if len(active_items) > 0:
            log_counts = torch.log(item_counts + 1)
            max_log = torch.max(log_counts)
            if max_log > 0:
                # Популярные имеют tau_min (жестче), редкие имеют tau_max (мягче)
                self.item_tau = self.tau_max - (self.tau_max - self.tau_min) * (log_counts / max_log)
            else:
                self.item_tau = torch.full_like(item_counts, self.tau_max)
        else:
            self.item_tau = torch.full_like(item_counts, self.tau_max)

        # -- Веса для Semantic Gating (Экспоненциальное затухание) --
        if self.use_semantic_gating and len(active_items) > 0:
            gating_tau = torch.median(active_items.float()).item()
            # У популярных gating_weight стремится к 0, у холодных стремится к gating_w_max
            self.gating_weights = self.gating_w_max * torch.exp(-item_counts / (gating_tau + 1e-5))
            self.gating_weights = self.gating_weights.unsqueeze(1)
        else:
            self.gating_weights = torch.zeros(self.n_items, 1, device=self.device)

    def forward(self):
        """
        Прямой проход: Классическая графовая свертка + Семантическое гейтирование.
        """
        # 1. Получаем классические CF эмбеддинги из графа
        user_all_embeddings, item_all_embeddings_cf = super().forward()

        # 2. Если включено гейтирование, подмешиваем семантику
        if self.use_semantic_gating and self.raw_semantic_embs is not None:
            # Получаем проецированную семантику для всех фильмов: (n_items, latent_dim)
            item_sem_embeddings = self.semantic_gating_proj(self.raw_semantic_embs)
            
            # Смешиваем: (1 - w) * E_cf + w * E_sem
            item_all_embeddings_final = (1 - self.gating_weights) * item_all_embeddings_cf + self.gating_weights * item_sem_embeddings
            
            # Возвращаем смешанные эмбеддинги. 
            # Для лосса и метрик будут использоваться именно они, что решает проблему холодного старта "из коробки"
            return user_all_embeddings, item_all_embeddings_final
        
        return user_all_embeddings, item_all_embeddings_cf

    def calculate_loss(self, interaction):
        """
        Вычисление общего лосса (BPR + Semantic SSL Loss) с автоматической балансировкой.
        """
        # Предотвращаем кэширование старых эмбеддингов между батчами из базового LightGCN
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # --- 1. Классический Collaborative Filtering (BPR Loss) ---
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # L2 Регуляризация эмбеддингов 0-го слоя
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings,
            require_pow=getattr(self, 'require_pow', False)
        )

        loss_bpr = mf_loss + self.reg_weight * reg_loss

        # Если семантика не используется, возвращаем только BPR
        if self.semantic_centroids is None or not self.use_semantic_loss:
            return loss_bpr

        # --- 2. Adaptive Semantic Contrastive Loss (SSL) ---
        cluster_ids = self.item2cluster[pos_item]
        
        # EMA (Exponential Moving Average) обновление динамических маяков:
        # Динамические маяки сдвигаются к графовым представлениям популярных фильмов.
        if self.training and self.use_centroid_ema:
            with torch.no_grad():
                hit_mask_float = (self.item_counts[pos_item] >= self.hit_threshold).float().unsqueeze(1)
                valid_embeddings = pos_embeddings * hit_mask_float
                
                cluster_one_hot = F.one_hot(cluster_ids, num_classes=self.n_clusters).float()
                cluster_one_hot_hits = cluster_one_hot * hit_mask_float
                
                sum_embeddings = torch.matmul(cluster_one_hot_hits.t(), valid_embeddings)
                counts = cluster_one_hot_hits.sum(dim=0).unsqueeze(1)
                
                c_k_tilde = sum_embeddings / (counts + 1e-9)
                update_mask = (counts > 0).float()
                
                self.dynamic_centroids.copy_((
                    self.ema_momentum * self.dynamic_centroids + 
                    (1.0 - self.ema_momentum) * c_k_tilde
                ) * update_mask + self.dynamic_centroids * (1.0 - update_mask))
                        
        all_centroids_matrix = F.normalize(self.dynamic_centroids, dim=1)

        # Выбираем узлы для семантического сближения
        if self.semantic_sampling == 'uniform':
            batch_size = pos_item.shape[0]
            semantic_items = torch.randint(1, self.n_items, (batch_size,), device=self.device)
        else:
            semantic_items = pos_item
            batch_size = semantic_items.shape[0]

        semantic_cluster_ids = self.item2cluster[semantic_items]
        semantic_item_embs = F.normalize(item_all_embeddings[semantic_items], dim=1)

        # Косинусное сходство сэмплированных фильмов со всеми центроидами
        cos_sim = torch.matmul(semantic_item_embs, all_centroids_matrix.t())
        
        # Dead-Zone Margin: отключаем притяжение, если вектор уже достаточно похож на свой кластер
        batch_indices = torch.arange(batch_size, device=self.device)
        pos_sims = cos_sim[batch_indices, semantic_cluster_ids]
        margin_mask = (pos_sims < self.semantic_margin).float()

        # Применяем пер-нодовые температуры (Degree-Aware Temperature)
        batch_tau = self.item_tau[semantic_items].unsqueeze(1)
        logits = cos_sim / batch_tau
        
        proto_loss_vector = F.cross_entropy(logits, semantic_cluster_ids, reduction='none')

        # Взвешиваем семантический лосс
        batch_alphas = self.item_alpha_weights[semantic_items]
        sem_loss_raw = (batch_alphas * proto_loss_vector * margin_mask).mean()

        # --- 3. Homoscedastic Uncertainty (Auto-balancing) ---
        # Авто-балансировка мультитаск-обучения (BPR vs Semantic лоссы) без жестких ручных весов.
        precision_bpr = torch.exp(-self.log_vars[0])
        loss_bpr_dynamic = precision_bpr * loss_bpr + self.log_vars[0]
        
        precision_sem = torch.exp(-self.log_vars[1])
        sem_loss_dynamic = precision_sem * sem_loss_raw + self.log_vars[1]
        
        total_dynamic_loss = loss_bpr_dynamic + sem_loss_dynamic

        # Случайное микро-логирование весов автобалансировки (примерно 1 раз за эпоху)
        if self.training and torch.rand(1).item() < 0.005:
            print(f"\nAUTO-BALANCE: BPR weight = {precision_bpr.item():.4f} | Sem weight = {precision_sem.item():.4f}")

        return total_dynamic_loss, loss_bpr, sem_loss_raw

    def predict(self, interaction):
        """Инференс для оценки пары пользователь-фильм (с калибровкой)."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e[item]

        # --- Fractional Magnitude Calibration ---
        # Считаем L2-норму и возводим ее в дробную степень для подавления популярных объектов
        i_norms = torch.norm(i_embeddings, p=2, dim=1)
        i_norms_gamma = torch.pow(i_norms, self.magnitude_gamma)
        i_embeddings_calibrated = i_embeddings / (i_norms_gamma.unsqueeze(1) + 1e-9)

        scores = torch.mul(u_embeddings, i_embeddings_calibrated).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Быстрый инференс для всех фильмов (используется для метрик Top-K)."""
        user = interaction[self.USER_ID]
        
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        item_all_embeddings = self.restore_item_e.clone()

        # --- Fractional Magnitude Calibration ---
        i_norms = torch.norm(item_all_embeddings, p=2, dim=1)
        i_norms_gamma = torch.pow(i_norms, self.magnitude_gamma)
        item_all_embeddings_calibrated = item_all_embeddings / (i_norms_gamma.unsqueeze(1) + 1e-9)

        scores = torch.matmul(u_embeddings, item_all_embeddings_calibrated.transpose(0, 1))
        return scores.view(-1)
