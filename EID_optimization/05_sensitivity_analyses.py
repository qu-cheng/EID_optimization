import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, make_scorer
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def define_feature_categories():
    categories = {
        'Selection Dynamics': ['neighbor_selected_ratio','synergy_score','redundancy_score','new_coverage_ratio'],
        'Global topology': ['density', 'avg_degree', 'degree_skewness'],
        'Global probability': ['prob_mean'],
        'Node topology': ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'local_density'],
        'Node probability & topology': ['prob_weighted_degree'],
        'Node probability': ['probability'],
    }
    return categories

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
            k = max(1, int(0.3 * n))  # top 30%
            ndcg_val = ndcg_score(
                [g['y_true'].values],
                [g['y_pred'].values],
                k=k
            )
            ndcgs.append(ndcg_val)

        return np.mean(ndcgs) if ndcgs else 0.0

ndcg_scorer = make_scorer(ndcg_grouped_score, greater_is_better=True)

def get_relevant_columns(df):
    categories = define_feature_categories()
    
    all_features = []
    for features in categories.values():
        all_features.extend(features)
    
    relevant_cols = ['rank'] + all_features
    
    available_cols = [col for col in relevant_cols if col in df.columns]
    
    return available_cols

def calculate_permutation_importance(model, X, y, n_repeats=100, random_state=42):
    perm_importance = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_state,
        n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'permutation_importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('permutation_importance', ascending=False)
    
    return perm_importance_df

def calculate_group_permutation_importance(model, X, y, categories, metric=ndcg_score, n_repeats=100, random_state=42):
    rng = np.random.RandomState(random_state)
    
    baseline_score = metric(y, model.predict(X))
    
    category_importance = {}
    
    for category, features in categories.items():
        available_features = [f for f in features if f in X.columns]
        if not available_features:
            category_importance[category] = {
                'importance_mean': 0.0,
                'importance_std': 0.0,
                'features': []
            }
            continue
        
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            for f in available_features:
                X_permuted[f] = rng.permutation(X_permuted[f].values)
            
            perm_score = metric(y, model.predict(X_permuted))
            scores.append(baseline_score - perm_score)
        
        category_importance[category] = {
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores),
            'features': available_features
        }
    
    return category_importance

# Rank_Permutation Importance
def calculate_rank_specific_importance(df, X_cols, y_col='ranking', n_ranks=8, n_repeats=10):
    rank_importance_dict = {}
    rank_category_importance_dict = {}
    categories = define_feature_categories()

    dynamic_features = categories['Selection Dynamics']

    for rank in range(n_ranks):
        df_rank = df[df[y_col] >= rank].copy()
        y_rank = (df_rank[y_col] == rank).astype(int)

        if rank == 0:
            allowed_features = [f for f in X_cols if f not in dynamic_features and f in df_rank.columns]
        else:
            allowed_features = [f for f in X_cols if f in df_rank.columns]

        X_rank = df_rank[allowed_features]

        if len(X_rank) == 0 or y_rank.sum() == 0:
            empty_df = pd.DataFrame({
                'feature': allowed_features,
                'importance': [0.0] * len(allowed_features),
                'importance_norm': [0.0] * len(allowed_features)
            })
            rank_importance_dict[rank] = empty_df
            category_importance = {category: 0.0 for category in categories.keys()}
            rank_category_importance_dict[rank] = category_importance
            continue

        # rank-specific random forest model
        rf = RandomForestRegressor(
            n_estimators=150, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_rank, y_rank)

        # Calculate Permutation Importance
        perm_result = permutation_importance(
            rf, X_rank, y_rank,
            n_repeats=n_repeats,
            n_jobs=-1,
            random_state=42
        )

        perm_df = pd.DataFrame({
            'feature': allowed_features,
            'importance': perm_result.importances_mean
        })

        # Normalize importance
        total_importance = perm_df['importance'].sum()
        if total_importance > 0:
            perm_df['importance_norm'] = perm_df['importance'] / total_importance
        else:
            perm_df['importance_norm'] = 0.0
            
        rank_importance_dict[rank] = perm_df[['feature', 'importance', 'importance_norm']]

        # grouped importance
        category_importance = {}
        for category, features in categories.items():
            category_features = [f for f in features if f in perm_df['feature'].values]
            if category_features:
                cat_score = perm_df[perm_df['feature'].isin(category_features)]['importance_norm'].sum()
                category_importance[category] = cat_score
            else:
                category_importance[category] = 0.0
        rank_category_importance_dict[rank] = category_importance

    return rank_importance_dict, rank_category_importance_dict

# Bayesian Optimization for Random Forest
def train_optimized_model(X_train, y_train):

    search_spaces = {
        'n_estimators': Integer(100, 200),
        'max_depth': Integer(8, 15),
        'min_samples_split': Integer(5, 15),
        'min_samples_leaf': Integer(3, 10),
        'max_features': Real(0.4, 0.8),
        'max_samples': Real(0.6, 0.9),
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=4)
    
    bayes_search = BayesSearchCV(
        estimator=rf,
        search_spaces=search_spaces,
        n_iter=30,
        scoring=ndcg_scorer,
        n_jobs=4,
        cv=5,
        random_state=42,
        verbose=0
    )
    
    bayes_search.fit(X_train, y_train)
    best_params = bayes_search.best_params_

    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    
    return final_model, best_params

def process_single_file(file_path, output_dir):

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        df = pd.read_csv(file_path)
        relevant_cols = get_relevant_columns(df)
        df_filtered = df[relevant_cols].copy()
        if 'ranking' not in df_filtered.columns:
            print(f" Warning: the file {file_name} does not contain 'ranking' column, skipping...")
            return
        X = df_filtered.drop(['ranking'], axis=1)
        y = df_filtered['ranking']
        
        if X.empty:
            print(f"  Warning: the file {file_name} has no valid features after filtering, skipping...")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        final_model, best_params = train_optimized_model(X_train, y_train)
        final_model.fit(X, y)
        
        # 1. Calculate permutation importance
        perm_importance_df = calculate_permutation_importance(
            final_model, X_test, y_test, n_repeats=50, random_state=42
        )
        perm_output_path = os.path.join(output_dir, f"{file_name}_feature_importance.csv")
        perm_importance_df.to_csv(perm_output_path, index=False)
        
        # 2. Calculate overall model category importance
        categories = define_feature_categories()
        category_importance = calculate_group_permutation_importance(
            final_model, X_test, y_test, 
            categories=categories,
            n_repeats=50, random_state=42
        )
        
        category_df = pd.DataFrame([
            {
                'category': category,
                'importance_mean': data['importance_mean'],
                'importance_std': data['importance_std'],
                'num_features': len(data['features'])
            }
            for category, data in category_importance.items()
        ]).sort_values('importance_mean', ascending=False)
        
        category_output_path = os.path.join(output_dir, f"{file_name}_category_importance.csv")
        category_df.to_csv(category_output_path, index=False)
        
        # 3. Calculate rank-specific importance
        rank_importance_dict, rank_category_importance_dict = calculate_rank_specific_importance(
            df_filtered, X.columns, y_col='ranking', n_ranks=10, n_repeats=50
        )
        
        rank_feature_dfs = []
        for rank, rank_df in rank_importance_dict.items():
            rank_df_copy = rank_df.copy()
            rank_df_copy['rank'] = rank
            rank_feature_dfs.append(rank_df_copy)
        
        if rank_feature_dfs:
            combined_rank_feature_df = pd.concat(rank_feature_dfs, ignore_index=True)
            rank_feature_output_path = os.path.join(output_dir, f"{file_name}_rank_feature_importance.csv")
            combined_rank_feature_df.to_csv(rank_feature_output_path, index=False)
        
        rank_category_df = pd.DataFrame(rank_category_importance_dict).T
        rank_category_df.index.name = 'rank'
        rank_category_output_path = os.path.join(output_dir, f"{file_name}_rank_category_importance.csv")
        rank_category_df.to_csv(rank_category_output_path)
        
    except Exception as e:
        print(f"  Error: {str(e)}")

def batch_process_files(input_dir, output_dir, file_pattern="*.csv"):
    os.makedirs(output_dir, exist_ok=True)
    search_pattern = os.path.join(input_dir, file_pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        return

    for i, file_path in enumerate(files, 1):
        process_single_file(file_path, output_dir)


def main():
    input_directory = r"D:\\data_generating\\datasets_split"
    output_directory = r"D:\\data_generating\\importance"
    batch_process_files(input_directory, output_directory, "*.csv")
    print("  1. {filename}_feature_importance.csv")
    print("  2. {filename}_category_importance.csv ") 
    print("  3. {filename}_rank_feature_importance.csv ")
    print("  4. {filename}_rank_category_importance.csv")

if __name__ == "__main__":
    main()