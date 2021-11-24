param_space_lr = {

}

param_space_lasso = {
    'alpha': [1, 0.1, 0.01, 0.001],
    'random_state': [20211014]
}

param_space_rf = {
    'max_depth': [4, 6, 8],
    # 'min_child_weight': [1, 5, 15],
    'min_samples_split': [2, 4, 8],
    # 'min_samples_leaf': [1, 2, 4],
    # 'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 500],
    'random_state': [20211014]

}

param_space_xgb = {
    'max_depth': [4, 6, 8],
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.1, 0.05, 0.01],
    'seed': [20211014]
}