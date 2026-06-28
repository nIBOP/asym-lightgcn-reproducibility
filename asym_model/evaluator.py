import torch
import pandas as pd
import math
from recbole.utils.case_study import full_sort_topk

def dcg_at_k(recs, relevant_set):
    return sum(
        (1 / math.log2(idx + 2)) if item in relevant_set else 0
        for idx, item in enumerate(recs)
    )

def evaluate_stratified(trainer, test_data, train_data, k=10):
    model = trainer.model
    print('\n' + '='*50)
    print('СТРАТИФИЦИРОВАННЫЙ АНАЛИЗ (HEAD / NON-HEAD)')

    device = model.device

    # 1. Извлекаем и считаем популярность всех фильмов из обучающего графа
    dataset = train_data.dataset
    iid_field = dataset.iid_field
    uid_field = dataset.uid_field

    train_item_ids = dataset.inter_feat[iid_field].numpy()
    train_item_counts = pd.Series(train_item_ids).value_counts()
    
    test_user_ids_arr = test_data.dataset.inter_feat[uid_field].numpy()
    test_item_ids_arr = test_data.dataset.inter_feat[iid_field].numpy()

    total_items = len(train_item_counts)

    # 2. Задаем границы корзин (20% / 30% / 50% по числу уникальных объектов)
    # ВАЖНО: Стратификация делается по числу уникальных объектов, 
    # а не по кумулятивной массе взаимодействий.
    head_cutoff = max(1, int(total_items * 0.20))
    torso_cutoff = max(head_cutoff + 1, int(total_items * 0.50)) # 20% + 30%
    torso_cutoff = min(torso_cutoff, total_items)

    # 3. Разбиваем на множества (set) для быстрого поиска
    head_items = set(train_item_counts.index[:head_cutoff])
    torso_items = set(train_item_counts.index[head_cutoff:torso_cutoff])
    tail_items_train = set(train_item_counts.index[torso_cutoff:])
    
    # ХВОСТ: Редкие фильмы (из train) + ВСЕ холодные (Cold-start) фильмы, которые реально есть в test, но нет в train
    test_items = set(test_item_ids_arr)
    cold_start_items = test_items - set(train_item_counts.index)
    tail_items = tail_items_train | cold_start_items

    non_head_items = torso_items | tail_items

    print(f'Распределение базы:')
    print(f'   - Head (Топ 20%): {len(head_items)} элементов')
    print(f'   - Torso (следующие 30%): {len(torso_items)} элементов')
    print(f'   - Tail (остальные 50% + Cold Start): {len(tail_items)} элементов')
    print(f'   - Non-head (Torso + Tail): {len(non_head_items)} элементов')

    # 4. Оценка
    user_recall_nh = []
    user_ndcg_nh = []
    user_recall_torso = []
    user_ndcg_torso = []
    user_recall_tail = []
    user_ndcg_tail = []
    user_recall_head = []
    user_recall = []
    user_ndcg = []
    
    num_recall_nh = 0
    den_recall_nh = 0
    
    num_non_head_share = 0
    den_non_head_share = 0
    
    num_recall_micro = 0
    den_recall_micro = 0

    # Выделяем уникальных пользователей из теста
    target_users = pd.unique(test_user_ids_arr)

    print(f'\nОценка на {len(target_users)} пользователях из теста...')

    import tqdm
    for uid in tqdm.tqdm(target_users, desc='Metrics Evaluation'):
        # Получаем Ground Truth для пользователя
        user_mask = (test_user_ids_arr == uid)
        user_test_items = test_item_ids_arr[user_mask]

        if len(user_test_items) == 0: continue
        
        T = set(user_test_items)

        # Получаем предсказания модели
        # Tensor обязательно должен быть на CPU перед full_sort_topk, 
        # иначе сломается индексация истории внутри RecBole (history_item)
        uid_tensor = torch.tensor([uid], dtype=torch.long)
        _, topk_tensor = full_sort_topk(uid_tensor, model, test_data, k=k, device=device)
        R = topk_tensor[0].cpu().numpy().tolist()
        R_set = set(R)

        # Non-head metrics
        T_nh = T & non_head_items
        
        if len(T_nh) > 0:
            user_recall_nh.append(len(R_set & T_nh) / len(T_nh))
            
            dcg = dcg_at_k(R, T_nh)
            ideal_rels = min(len(T_nh), len(R))
            idcg = sum(1 / math.log2(i + 2) for i in range(ideal_rels))
            user_ndcg_nh.append(dcg / idcg if idcg > 0 else 0)
            
        num_recall_nh += len(R_set & T_nh)
        den_recall_nh += len(T_nh)
        
        # NonHeadShare@K
        num_non_head_share += len(R_set & non_head_items)
        den_non_head_share += len(R)
        
        # Torso / Tail metrics
        T_torso = T & torso_items
        if len(T_torso) > 0:
            user_recall_torso.append(len(R_set & T_torso) / len(T_torso))
            
            dcg_torso = dcg_at_k(R, T_torso)
            ideal_rels_torso = min(len(T_torso), len(R))
            idcg_torso = sum(1 / math.log2(i + 2) for i in range(ideal_rels_torso))
            user_ndcg_torso.append(dcg_torso / idcg_torso if idcg_torso > 0 else 0)
            
        T_tail = T & tail_items
        if len(T_tail) > 0:
            user_recall_tail.append(len(R_set & T_tail) / len(T_tail))
            
            dcg_tail = dcg_at_k(R, T_tail)
            ideal_rels_tail = min(len(T_tail), len(R))
            idcg_tail = sum(1 / math.log2(i + 2) for i in range(ideal_rels_tail))
            user_ndcg_tail.append(dcg_tail / idcg_tail if idcg_tail > 0 else 0)
        
        # Head metrics
        T_head = T & head_items
        if len(T_head) > 0:
            user_recall_head.append(len(R_set & T_head) / len(T_head))
            
        # Overall metrics
        user_recall.append(len(R_set & T) / len(T))
        
        num_recall_micro += len(R_set & T)
        den_recall_micro += len(T)
        
        dcg_overall = dcg_at_k(R, T)
        ideal_rels_overall = min(len(T), len(R))
        idcg_overall = sum(1 / math.log2(i + 2) for i in range(ideal_rels_overall))
        user_ndcg.append(dcg_overall / idcg_overall if idcg_overall > 0 else 0)

    # 5. Вывод результатов
    recall_nh = sum(user_recall_nh) / len(user_recall_nh) if len(user_recall_nh) > 0 else 0
    recall_nh_micro = num_recall_nh / den_recall_nh if den_recall_nh > 0 else 0
    ndcg_nh = sum(user_ndcg_nh) / len(user_ndcg_nh) if len(user_ndcg_nh) > 0 else 0
    
    non_head_share = num_non_head_share / den_non_head_share if den_non_head_share > 0 else 0
    
    recall_torso = sum(user_recall_torso) / len(user_recall_torso) if len(user_recall_torso) > 0 else 0
    ndcg_torso = sum(user_ndcg_torso) / len(user_ndcg_torso) if len(user_ndcg_torso) > 0 else 0
    
    recall_tail = sum(user_recall_tail) / len(user_recall_tail) if len(user_recall_tail) > 0 else 0
    ndcg_tail = sum(user_ndcg_tail) / len(user_ndcg_tail) if len(user_ndcg_tail) > 0 else 0

    recall_head = sum(user_recall_head) / len(user_recall_head) if len(user_recall_head) > 0 else 0
    
    recall = sum(user_recall) / len(user_recall) if len(user_recall) > 0 else 0
    recall_micro = num_recall_micro / den_recall_micro if den_recall_micro > 0 else 0
    ndcg = sum(user_ndcg) / len(user_ndcg) if len(user_ndcg) > 0 else 0
    
    results = {
        'recall_nh': recall_nh,
        'recall_nh_micro': recall_nh_micro,
        'ndcg_nh': ndcg_nh,
        'non_head_share': non_head_share,
        'recall_torso': recall_torso,
        'ndcg_torso': ndcg_torso,
        'recall_tail': recall_tail,
        'ndcg_tail': ndcg_tail,
        'recall_head': recall_head,
        'recall': recall,
        'recall_micro': recall_micro,
        'ndcg': ndcg
    }
    
    print(f'\nРезультаты (K={k}):')
    for metric_name, v in results.items():
        print(f'   - {metric_name}: {v:.4f}')

    print('='*50 + '\n')
    return results
