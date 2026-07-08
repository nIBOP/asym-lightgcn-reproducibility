# `AsymLightGCN`: описание текущей реализации

Этот файл описывает не абстрактную идею модели, а именно ту реализацию, которая сейчас лежит в репозитории:

- модель: [`asym_model/model.py`](./model.py)
- тренер: [`asym_model/trainer_compact.py`](./trainer_compact.py)
- стратифицированная оценка: [`asym_model/evaluator.py`](./evaluator.py)
- основной запуск: [`run_final_full_graph_benchmarks.py`](../run_final_full_graph_benchmarks.py)
- конфиг: [`config.yaml`](../config.yaml)

Отдельная `hero`-модель и отдельный `hero`-конфиг в этом пакете не используются. Описание ниже опирается на фактическую схему запуска `AsymLightGCN` через текущий `config.yaml` и на дефолты, которые RecBole подтягивает для `LightGCN`.

---

## 1. Что это за модель

`AsymLightGCN` наследуется от `recbole.model.general_recommender.lightgcn.LightGCN` и добавляет поверх базового графового CF пять механизмов:

1. **Semantic Gating**: подмешивает текстовую семантику в item-эмбеддинги, особенно для редких и cold-start объектов.
2. **Semantic Prototype Loss**: притягивает item-эмбеддинги к семантическим центроидам кластеров.
3. **Dynamic Centroids**: обновляет центроиды EMA-правилом, используя только достаточно популярные объекты.
4. **Asymmetric degree-aware regularization**: делает температуру и вес semantic loss зависимыми от популярности item.
5. **Fractional Magnitude Calibration**: на инференсе подавляет переоценку хитов через норму item-вектора.

Цель архитектуры: одновременно ослабить `cold-start` и `popularity bias`, не теряя силу обычного `LightGCN` на head-объектах.

---

## 2. Как модель встроена в пайплайн

Есть важная техническая деталь: в RecBole мы по-прежнему создаем `Config(model='LightGCN', ...)`, а уже потом вручную инстанцируем `AsymLightGCN`.

Это означает:

- все базовые дефолты тянутся из конфигурации `LightGCN`;
- поверх них код сам добавляет наши дополнительные поля из `config.yaml`;
- формально в конфиге модель называется `LightGCN`, но фактически обучается `AsymLightGCN`.

Текущие базовые значения, которые RecBole подтягивает для запуска из `config.yaml`:

| Параметр | Значение |
| --- | --- |
| `embedding_size` | `64` |
| `n_layers` | `2` |
| `reg_weight` | `1e-5` |
| `require_pow` | `False` |
| negative sampling | uniform, `1` negative per positive |
| `epochs` | `300` |
| `train_batch_size` | `2048` |
| `learning_rate` | `0.001` |
| `stopping_step` | `10` |
| validation metric | `NDCG@10` |
| metrics | `Recall`, `NDCG` |
| `topk` | `10`, `50` |
| split | `0.8 / 0.1 / 0.1` |
| eval mode | full ranking |

Иными словами: основа у нас все еще классический `LightGCN`, а вся асимметрия и семантика добавляются поверх него.

---

## 3. Какие внешние артефакты нужны модели

Модель опирается на три внешних семантических артефакта:

| Артефакт | Откуда берется | Для чего нужен |
| --- | --- | --- |
| `centroids_path` | `cluster_centroids.pt` после KMeans | базовые семантические центроиды |
| `item_mapping_path` | `.item` файл с `item_id:token` и `cluster_id:token` | связь item -> cluster |
| `semantic_embs_path` | `semantic_embeddings.pt` | сырые текстовые эмбеддинги item |

Для `clean_movies` они готовятся в [`dataset_prep/data.py`](../dataset_prep/data.py):

1. текст item кодируется через `SentenceTransformer('all-MiniLM-L6-v2')`;
2. полученные embedding'и кластеризуются `KMeans(n_clusters=150)`;
3. сохраняются:
   - центроиды;
   - словарь `item_id -> embedding`;
   - `.item` mapping с `cluster_id:token`;
   - `.inter` файл для RecBole.

Аналогичная логика есть для Amazon и Yelp в [`dataset_prep/data_amazon.py`](../dataset_prep/data_amazon.py) и [`dataset_prep/data_yelp.py`](../dataset_prep/data_yelp.py).

Практический нюанс: в репозитории есть дубли артефактов и по путям `dataset_prep/...`, и по путям `clean_movies/...`. Основной `config.yaml` для `clean_movies` смотрит на `dataset_prep/...`, а некоторые экспериментальные скрипты переопределяют пути на root-level каталоги.

---

## 4. Архитектура по шагам

### 4.1. Базовый графовый слой

`AsymLightGCN` сначала делает обычный проход `LightGCN`:

```text
(user_all_embeddings, item_all_embeddings_cf) = super().forward()
```

Это стандартные коллаборативные user/item-эмбеддинги после `n_layers=2` графовых агрегаций.

Если никакие наши надстройки не включены, модель сводится к LightGCN с BPR-обучением.

### 4.2. Загрузка и проекция семантических центроидов

При инициализации модель загружает `semantic_centroids` из `centroids_path`.

Если размерность центроидов не совпадает с `latent_dim`, включается проекционная MLP:

```text
Linear(centroid_dim -> 2 * latent_dim)
LayerNorm
LeakyReLU
Linear(2 * latent_dim -> latent_dim)
```

Это важно, потому что text-эмбеддинги Sentence-BERT и графовые эмбеддинги живут в разных пространствах. Для согласования пространств модель держит буфер `dynamic_centroids`, который инициализируется либо проекцией центроидов, либо их прямой копией.

### 4.3. Semantic Gating

Если `use_semantic_gating=True`, модель дополнительно загружает сырые текстовые item-эмбеддинги из `semantic_embs_path` и учит еще одну проекцию:

```text
Linear(raw_dim -> latent_dim)
LayerNorm
LeakyReLU
```

После этого для каждого item считается итоговый вектор:

```text
E_item_final = (1 - w_i) * E_item_cf + w_i * E_item_sem
```

где:

```text
w_i = gating_w_max * exp(-count_i / gating_tau)
gating_tau = median(active_item_counts)
```

Смысл механизма:

- у head-item `count_i` большой, значит `w_i` близок к нулю;
- у tail-item и cold-start item `w_i` высокий;
- поэтому редкие объекты получают больше текстовой поддержки, а популярные остаются в основном графовыми.

В текущем конфиге:

| Параметр | Значение | Смысл |
| --- | --- | --- |
| `use_semantic_gating` | `True` | semantic gating включен |
| `gating_w_max` | `0.8` | максимальная доля семантики для редких item |

### 4.4. Degree-aware параметры

Перед обучением модель считает `item_counts` по train interactions и на их основе строит три зависимости.

#### A. Порог хитов для обновления центроидов

```text
hit_threshold = quantile(active_item_counts, hit_quantile)
```

Только item с `count_i >= hit_threshold` участвуют в адаптации `dynamic_centroids`.

#### B. Индивидуальная температура `tau_i`

Температура для semantic loss зависит от популярности:

```text
tau_i = tau_max - (tau_max - tau_min) * log(count_i + 1) / max_log_count
```

Следствие:

- у популярных item температура ближе к `tau_min`, то есть распределение жестче;
- у редких item температура ближе к `tau_max`, то есть regularization мягче.

В текущем конфиге:

| Параметр | Значение |
| --- | --- |
| `tau_min` | `0.2` |
| `tau_max` | `0.8` |
| `hit_quantile` | `0.8` |

#### C. Индивидуальный вес semantic loss `alpha_i`

Логика зависит от `alpha_strategy`.

Поддерживаются три режима:

- `static`: все `alpha_i = 1`;
- `dynamic_leaky_log`: alpha снижается для super-head item после заданного квантиля;
- любой другой режим в коде фактически работает как `inverse_log`.

Формулы:

```text
static:
alpha_i = 1

dynamic_leaky_log:
cutoff = quantile(active_item_counts, dynamic_cutoff_quantile)
raw_alpha_i = 1 - log(count_i + 1) / log(cutoff + 1)
alpha_i = clamp(raw_alpha_i, semantic_weight_floor, semantic_weight_cap)

fallback / inverse_log:
raw_alpha_i = 1 / log(e + count_i)
alpha_i = clamp(raw_alpha_i, max=semantic_weight_cap)
```

В текущем конфиге стоит:

| Параметр | Значение |
| --- | --- |
| `alpha_strategy` | `static` |
| `semantic_weight_floor` | `0.05` |
| `semantic_weight_cap` | `0.1` |
| `dynamic_cutoff_quantile` | `0.85` |

То есть в базовом запуске асимметрия по `alpha` сейчас фактически выключена, а вот асимметрия по `tau` и по gating остается.

### 4.5. Semantic Prototype Loss

После вычисления BPR-loss модель считает дополнительный semantic loss.

Сначала для каждого positive item берется его cluster id:

```text
cluster_ids = item2cluster[pos_item]
```

Дальше есть два режима выборки объектов для semantic branch:

- `uniform`: случайные item из каталога;
- `batch`: те же `pos_item`, что и в текущем interaction batch.

В текущем конфиге используется:

| Параметр | Значение |
| --- | --- |
| `semantic_sampling` | `uniform` |

Дальше модель нормализует item-эмбеддинги и центроиды, считает косинусное сходство со всеми центроидами и превращает задачу в классификацию по прототипам:

```text
logits_i = cos(E_i, C_all) / tau_i
proto_loss_i = CrossEntropy(logits_i, true_cluster_i)
```

Это не классический pairwise contrastive loss item-item. В текущей реализации это именно **prototype classification loss**: правильным классом считается собственный semantic cluster item, а "негативами" выступают остальные центроиды.

### 4.6. Dead-zone margin

Чтобы не перетягивать item в центроид бесконечно, используется `semantic_margin`.

Если item уже достаточно близок к своему центроиду, его semantic loss зануляется:

```text
margin_mask_i = 1, если cos(E_i, C_true) < semantic_margin
margin_mask_i = 0, иначе
```

Итоговый raw semantic loss:

```text
L_sem_raw = mean(alpha_i * proto_loss_i * margin_mask_i)
```

В текущем конфиге:

| Параметр | Значение |
| --- | --- |
| `semantic_margin` | `0.8` |

### 4.7. Dynamic Centroids

Во время обучения `dynamic_centroids` адаптируются к графовому пространству.

Обновление идет только по hit-item текущего batch:

```text
c_tilde_k = average(pos_embeddings of hit items from cluster k)
C_k <- ema_momentum * C_k + (1 - ema_momentum) * c_tilde_k
```

Если в batch для кластера нет hit-item, его центроид не обновляется.

Зачем это нужно:

- стартовые центроиды рождаются из текста;
- графовое пространство со временем уезжает в сторону взаимодействий;
- EMA подтягивает центроиды ближе к реально обученному CF-пространству;
- это уменьшает разрыв между text-space и graph-space.

В текущем конфиге:

| Параметр | Значение |
| --- | --- |
| `ema_momentum` | `0.99` |

### 4.8. Основной BPR-loss

Коллаборативный лосс считается стандартно:

```text
pos_score = <u, i_pos>
neg_score = <u, i_neg>
L_bpr = BPR(pos_score, neg_score) + reg_weight * L2(ego_embeddings)
```

Здесь используются унаследованные из `LightGCN`:

- `mf_loss`;
- `reg_loss`;
- `user_embedding`;
- `item_embedding`.

Если semantic_centroids не удалось загрузить, модель возвращает только `L_bpr`, то есть training gracefully деградирует в обычный графовый CF.

### 4.9. Homoscedastic uncertainty balancing

Ручного коэффициента вроде старого `proto_reg_weight` в основной модели больше нет.

Вместо него модель учит два параметра:

```text
log_vars = nn.Parameter(torch.zeros(2))
```

И балансирует лоссы так:

```text
L_total =
exp(-s_bpr) * L_bpr + s_bpr +
exp(-s_sem) * L_sem_raw + s_sem
```

где `s_bpr = log_vars[0]`, `s_sem = log_vars[1]`.

Смысл:

- если semantic branch шумный, модель может автоматически уменьшить его вклад;
- если semantic loss полезен, вес branch вырастет сам;
- не нужно руками подбирать единый коэффициент под каждый датасет.

### 4.10. Fractional Magnitude Calibration на инференсе

На `predict()` и `full_sort_predict()` модель не использует item-векторы как есть, а калибрует их норму:

```text
score(u, i) = <u, i / ||i||^gamma>
```

или эквивалентно:

```text
i_calibrated = i / (||i||^gamma + eps)
score = <u, i_calibrated>
```

Если `gamma = 0`, получаем обычный dot product.

Если `gamma = 1`, получается почти cosine-like нормировка.

В текущем конфиге:

| Параметр | Значение |
| --- | --- |
| `magnitude_calibration_gamma` | `0.25` |

Это ключевой inference-time механизм против popularity bias: head-item часто выигрывают не только углом, но и длиной вектора, а gamma-калибровка эту "силу нормы" частично срезает.

---

## 5. Что происходит в `forward`, `calculate_loss`, `predict`

### `forward()`

1. строит графовые user/item embeddings через `LightGCN`;
2. при включенном gating строит semantic item embeddings;
3. смешивает CF и semantic item-векторы;
4. возвращает user-векторы и уже смешанные item-векторы.

Это значит, что semantic gating влияет не только на лосс, но и на downstream prediction.

### `calculate_loss()`

1. сбрасывает кэш `restore_user_e` и `restore_item_e`, чтобы не использовать старые embedding'и между batch'ами;
2. считает BPR-loss;
3. при наличии центроидов считает semantic prototype loss;
4. если идет training, обновляет dynamic centroids;
5. объединяет `L_bpr` и `L_sem_raw` через homoscedastic balancing;
6. возвращает кортеж:

```text
(total_dynamic_loss, loss_bpr, sem_loss_raw)
```

Именно поэтому `CustomTrainer` умеет логировать total/BPR/Sem отдельно.

### `predict()` и `full_sort_predict()`

Оба метода:

- используют уже готовые embedding'и из `forward()`;
- кэшируют их в `restore_user_e` и `restore_item_e`;
- перед скорингом выполняют magnitude calibration.

---

## 6. Разбор `config.yaml`

Ниже перечислены все параметры текущего конфига, которые реально используются `AsymLightGCN`.

### Данные и обучение

| Ключ | Значение | Назначение |
| --- | --- | --- |
| `data_path` | `./dataset_prep` | базовая папка датасета для `clean_movies` |
| `dataset` | `clean_movies` | имя датасета для RecBole |
| `USER_ID_FIELD` | `user_id` | поле пользователя |
| `ITEM_ID_FIELD` | `item_id` | поле item |
| `epochs` | `300` | максимум эпох |
| `train_batch_size` | `2048` | train batch size |
| `learning_rate` | `0.001` | шаг оптимизатора Adam |
| `eval_step` | `1` | валидация каждую эпоху |
| `stopping_step` | `10` | early stopping |
| `metrics` | `Recall`, `NDCG` | основные метрики |
| `topk` | `10`, `50` | глубины ранжирования |
| `valid_metric` | `NDCG@10` | метрика выбора лучшей эпохи |

### Семантические артефакты

| Ключ | Значение | Назначение |
| --- | --- | --- |
| `centroids_path` | `dataset_prep/cluster_centroids.pt` | стартовые центроиды |
| `item_mapping_path` | `dataset_prep/clean_movies/clean_movies.item` | item -> cluster mapping |
| `semantic_embs_path` | `dataset_prep/clean_movies/semantic_embeddings.pt` | сырые semantic embeddings |

### Semantic Gating

| Ключ | Значение | Назначение |
| --- | --- | --- |
| `use_semantic_gating` | `True` | включить подмешивание семантики |
| `gating_w_max` | `0.8` | верхняя граница веса semantic branch |

### Semantic loss и динамические центроиды

| Ключ | Значение | Назначение |
| --- | --- | --- |
| `tau_min` | `0.2` | минимальная температура, в основном для head-item |
| `tau_max` | `0.8` | максимальная температура, в основном для tail-item |
| `ema_momentum` | `0.99` | сглаживание для dynamic centroids |
| `hit_quantile` | `0.80` | кто считается hit-item для EMA update |
| `semantic_sampling` | `uniform` | как выбирать item для semantic branch |
| `semantic_margin` | `0.8` | dead-zone threshold |
| `alpha_strategy` | `static` | стратегия весов semantic loss |
| `semantic_weight_floor` | `0.05` | нижняя граница alpha при dynamic режиме |
| `semantic_weight_cap` | `0.1` | верхняя граница alpha при dynamic режиме |
| `dynamic_cutoff_quantile` | `0.85` | cutoff для `dynamic_leaky_log` |

### Инференс-калибровка

| Ключ | Значение | Назначение |
| --- | --- | --- |
| `magnitude_calibration_gamma` | `0.25` | сила подавления нормы item-вектора |

---

## 7. Что делает `CustomTrainer`

[`asym_model/trainer_compact.py`](./trainer_compact.py) меняет не математику модели, а operational behavior обучения.

Что он делает:

- логирует loss-компоненты каждые `50` batch'ей;
- копит loss как GPU tensor и синхронизирует CPU только в конце эпохи;
- печатает отдельно `Loss`, `BPR`, `Sem`;
- отключает стандартную `NaN`-проверку RecBole, чтобы не форсировать CPU-GPU sync на каждом шаге;
- вызывает `train_data.pr_end()` на последнем batch, чтобы dataloader корректно reshuffle/resample-ился.

Итог: модель обучается так же, но логирование и производительность лучше контролируются.

---

## 8. Как устроена стратифицированная оценка

[`asym_model/evaluator.py`](./evaluator.py) делает отдельный анализ качества по популярности item.

Разбиение строится по числу **уникальных item**, а не по массе взаимодействий:

- `Head`: топ-20% item по popularity в train;
- `Torso`: следующие 30%;
- `Tail`: оставшиеся 50% из train;
- `Cold-start`: item, которые есть в test, но отсутствуют в train;
- `Non-head`: `Torso + Tail + Cold-start`.

Считаются:

- `recall_nh`, `recall_nh_micro`, `ndcg_nh`;
- `non_head_share`;
- `recall_torso`, `ndcg_torso`;
- `recall_tail`, `ndcg_tail`;
- `recall_head`;
- общий `recall`, `recall_micro`, `ndcg`.

Это важная часть всей идеи модели: она оптимизируется не только под глобальный `Recall/NDCG`, но и под качество в non-head зоне.

---

## 9. Что именно отключают абляции

В репозитории есть несколько абляционных сценариев. Их полезно понимать буквально, по коду.

| Скрипт | Что реально отключает |
| --- | --- |
| `run_ablation_no_gating.py` | semantic gating |
| `run_ablation_no_calibration.py` | magnitude calibration на инференсе |
| `run_ablation_no_homoscedastic.py` | обучаемый balancing через `log_vars` |
| `run_ablation_static_centroids.py` | EMA-обновление dynamic centroids |
| `run_ablation_const_tau.py` | degree-aware `tau_i`, заменяя его константой `0.2` |
| `run_ablation_fixed_alpha_sweep.py` | фиксирует общий `alpha` и отключает homoscedastic balancing |
| `run_ablation_wo_asymmetry.py` | делает `item_alpha_weights` одинаковыми для всех item |
| `run_ablation_gamma_sweep.py` | меняет `gamma` только на инференсе без переобучения |

Важно: `run_ablation_wo_asymmetry.py` в текущем коде **не** убирает всю асимметрию модели. Он лишь убирает popularity-dependent alpha weighting. Асимметрия через `tau_i`, gating и gamma-калибровку при этом остается.

Еще один важный момент: `run_experiment_lambda.py` продолжает варьировать `proto_reg_weight`, но в текущей реализации `AsymLightGCN` этот коэффициент больше не используется. Для нынешней версии модели этот эксперимент отражает legacy-логику, а не текущий training objective.

---

## 10. Сценарии запуска

Основные entry points:

- [`run_final_full_graph_benchmarks.py`](../run_final_full_graph_benchmarks.py): поддерживаемый финальный full-graph benchmark на `clean_movies`, `clean_amazon` и `clean_yelp`;
- [`validate_final_full_graph_setup.py`](../validate_final_full_graph_setup.py): smoke-check данных, путей и инициализации моделей;
- legacy-эксперименты, абляции и sweep-сценарии перенесены в [`scripts/legacy/`](../scripts/legacy/README.md).

Для Amazon и Yelp скрипты переопределяют пути к semantic artifacts через `config_dict`, потому что там используются отдельные наборы файлов.

---

## 11. Ограничения и честные caveats реализации

1. Если `semantic_centroids` не загрузились, semantic loss отключится, но BPR-обучение продолжится.
2. Если не загрузились raw semantic embeddings, semantic gating автоматически выключится.
3. `item2cluster` инициализируется нулями. Значит unmapped item попадут в cluster `0`, если mapping неполный.
4. Для item без semantic embedding в `raw_semantic_embs` останется нулевой вектор.
5. В базовом `config.yaml` стоит `alpha_strategy='static'`, поэтому часть "асимметрии" сейчас задается не через alpha, а через gating, `tau_i` и gamma-калибровку.
6. Некоторые legacy-скрипты в `scripts/legacy/experiments/`, включая старый `main.py`, чистят `.pth`-файлы через `clean_cache()`, поэтому хранить важные артефакты в неожиданных root-level `.pth` без дублирования рискованно.
7. [`asym_model/__init__.py`](./__init__.py) глобально патчит `torch.load`, принудительно выставляя `weights_only=False`. Это помогает грузить старые `.pt`-артефакты со словарями, но является глобальным side effect для всего процесса.

---

## 12. Коротко: итоговая формула модели

На практике текущая модель работает так:

1. `LightGCN` строит CF-эмбеддинги.
2. Для редких item вектор усиливается текстовой семантикой через gating.
3. Item дополнительно учатся предсказывать свой semantic prototype среди всех центроидов.
4. Центроиды медленно подстраиваются под графовое пространство через EMA.
5. Баланс BPR и semantic loss учится автоматически.
6. На выдаче score head-item уменьшается через дробную нормировку `||i||^gamma`.

Если совсем коротко, `AsymLightGCN` в этом репозитории это:

```text
LightGCN
+ popularity-aware semantic gating
+ prototype-based semantic regularization
+ dynamic centroids
+ uncertainty-balanced multitask loss
+ magnitude calibration at inference
```
