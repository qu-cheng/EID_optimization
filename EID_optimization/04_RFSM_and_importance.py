import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import ndcg_score, make_scorer
from scipy.stats import spearmanr, kendalltau
import joblib
import math
import os
import matplotlib.pyplot as plt

rank_thresholds = [0.1, 0.2, 0.3]

for rank_threshold in rank_thresholds:
    base_dir = rf"D:\data_generating\performance_{rank_threshold}"
    os.makedirs(base_dir, exist_ok=True)
    def ndcg_grouped_score(y_true, y_pred, groups=None, **kwargs):
        df_temp = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': groups
        })

        ndcgs = []
        for gid, g in df_temp.groupby('group'):
            n = len(g)
            if n == 0:
                continue
            k = max(1, int(rank_threshold * n))
            ndcg_val = ndcg_score(
                [g['y_true'].values],
                [g['y_pred'].values],
                k=k
            )
            ndcgs.append(ndcg_val)

        return np.mean(ndcgs) if ndcgs else 0.0

    ndcg_scorer = make_scorer(ndcg_grouped_score, greater_is_better=True)

    file_path = r"D:\data_generating\training_data_1000.csv"
    df = pd.read_csv(file_path)
    df_below = df[df['rank'] <= rank_threshold].copy()
    df_above = df[df['rank'] > rank_threshold].copy()

    if len(df_above) > 0:
        df_above_sampled = df_above.sample(frac=1, random_state=42).copy()
        df_above_sampled['rank'] = rank_threshold
    else:
        df_above_sampled = df_above.copy()

    df_aug = pd.concat([df_below, df_above_sampled], ignore_index=True)

    feature_cols = [c for c in df_aug.columns
                    if c not in {'ranking', 'rank', 'node_id', 'network_id', 'num_nodes'}]
    X = df_aug[feature_cols]
    y_rank = df_aug['rank']
    groups = df_aug['network_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_rank, groups=groups))
    df_aug['split'] = 'none'
    df_aug.loc[train_idx, 'split'] = 'train'
    df_aug.loc[test_idx, 'split'] = 'test'

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_rank.iloc[train_idx], y_rank.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    cv_splitter = GroupKFold(n_splits=5)
    search_spaces = {
        'n_estimators': (100, 500),
        'max_depth': (5, 30),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5),
        'max_features': (0.3, 1.0),
        'max_samples': (0.5, 1.0)
    }
    base_rf = RandomForestRegressor(oob_score=False, n_jobs=-1, random_state=42)
    bayes_cv = BayesSearchCV(
        base_rf,
        search_spaces,
        n_iter=30,
        cv=cv_splitter,
        scoring=ndcg_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    bayes_cv.fit(X_train, y_train, groups=groups_train)
    best_rf = bayes_cv.best_estimator_

    best_rf.fit(X_train, y_train)
    df_aug['pred_rank'] = best_rf.predict(X)

    def calc_grouped_metrics(df_eval):
        results = []
        for gid, g in df_eval.groupby('network_id'):
            n = len(g)
            if n == 0:
                continue
            k = max(1, int(rank_threshold * n))
            y_true, y_pred = g['rank'].values, g['pred_rank'].values
            rho, _ = spearmanr(y_true, y_pred)
            tau, _ = kendalltau(y_true, y_pred)
            ndcg_val = ndcg_score([y_true], [y_pred], k=k)
            results.append((rho, tau, ndcg_val))
        return pd.DataFrame(results, columns=['Spearman', 'Kendall', 'NDCG']).mean().to_dict()

    train_metrics = calc_grouped_metrics(df_aug[df_aug['split'] == 'train'])
    test_metrics = calc_grouped_metrics(df_aug[df_aug['split'] == 'test'])

    print("\nTraining set metrics:", train_metrics)
    print("Test set metrics:", test_metrics)
    perm = permutation_importance(
        best_rf, X, y_rank,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring=ndcg_scorer
    )
    perm_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std
    }).sort_values('importance_mean', ascending=False)

    perm_path = os.path.join(base_dir, 'perm_importance_bayes.csv')
    perm_df.to_csv(perm_path, index=False)
    model_path = os.path.join(base_dir, 'rank_rf_bayes.pkl')
    joblib.dump(best_rf, model_path)