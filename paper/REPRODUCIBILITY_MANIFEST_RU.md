# Манифест воспроизводимости

Этот файл фиксирует, какие материалы должны сопровождать статью после правок рецензента. Его можно использовать как основу для README в публичном репозитории или Zenodo-архиве.

## Основной репозиторий

Принятая стратегия: отдельный публичный репозиторий `asym-lightgcn-reproducibility` в аккаунте `nIBOP`, лицензия кода MIT, затем при необходимости Zenodo DOI.

Планируемый публичный URL: https://github.com/nIBOP/asym-lightgcn-reproducibility

Локальная подготовленная папка: `paper/articles/evaluation_protocol/public_repo/asym-lightgcn-reproducibility`.

Перед отправкой статьи нужно создать публичный репозиторий, проверить, что ссылка открывается без авторизации, и при необходимости заменить ссылку в статье на Zenodo DOI.

## Код

- `asym_model/model.py` - реализация AsymLightGCN.
- `final_full_graph_config.py` - спецификации датасетов, путей к split-файлам, центроидам и семантическим эмбеддингам.
- `run_final_full_graph_benchmarks.py` - основной запуск графовых моделей.
- `run_reduced_extra_baselines.py` - дополнительные baseline/control-запуски.
- `prepare_paper_artifacts.py` - подготовка бумажных таблиц и артефактов.
- `evaluate_user_level_significance.py` - пользовательские bootstrap- и непараметрические проверки.
- `analyze_paper_significance.py` - парные seed-level сравнения.

## Конфиги и параметры

Ключевые параметры, отраженные в статье:

- split: случайное \(80/10/10\);
- metrics: Recall@10, NDCG@10, Recall@50, NDCG@50;
- valid metric: NDCG@10;
- optimizer: Adam;
- learning rate: 0,001;
- negative sampling: 1 отрицательный объект на положительный;
- embedding size: 64;
- AsymLightGCN layers: 2;
- regularization: \(10^{-5}\);
- reduced-срезы: 90 эпох, train/eval batch size 8192, eval step 15, early stopping 2.

## Данные и ограничения распространения

Исходные наборы данных не следует включать в архив, если их лицензии не разрешают повторное распространение. Вместо этого нужно приложить:

- ссылки на исходные страницы Amazon Books Reviews, The Movies Dataset и Yelp Open Dataset;
- скрипты подготовки срезов;
- split-файлы или инструкции их восстановления;
- контрольные статистики срезов: число пользователей, объектов, взаимодействий, среднее и медиана взаимодействий на пользователя;
- контрольные значения метрик из статьи.

## Семантические признаки

В архив должны входить:

- скрипты построения текстовых описаний объектов;
- указание модели `sentence-transformers/all-MiniLM-L6-v2`;
- скрипты построения `semantic_embeddings.pt`;
- скрипты построения семантических центроидов;
- mapping-файлы объектов RecBole.

## Статистика

Рекомендуется приложить файлы:

- `train_logs/paper_significance_2026-05-08.md`;
- `train_logs/paper_significance_2026-05-08.csv`;
- `train_logs/user_level_amazon_min5_2026-05-14.md`;
- `train_logs/user_level_amazon_min5_2026-05-14_paired_tests.csv`;
- `train_logs/amazon_threshold_user_level_2026-05-14.md`;
- `train_logs/amazon_threshold_user_level_2026-05-14_paired_tests.csv`.

## Что проверить перед подачей

- Репозиторий `https://github.com/nIBOP/asym-lightgcn-reproducibility` или Zenodo-архив публично открывается без авторизации.
- В архиве нет файлов, которые нельзя распространять.
- Пути в README совпадают с путями в статье.
- В статье указан актуальный URL или DOI.
- Контрольные статистики срезов совпадают с таблицами статьи.
