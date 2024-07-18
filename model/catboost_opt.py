import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import optuna
import numpy as np

RANDOM_SEED = 1

def catboost_optuna_objective(trial : optuna.Trial, dataset: cb.Pool, fixed_params: dict):
    scores = cb.cv(
        pool=dataset,
        params={
            **fixed_params,
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            # Maximum tree depth is 16
            'depth': trial.suggest_int('depth', 6, 16),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 0.1, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.75, 1),
            # for imbalanced datasets
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
            # 'iterations' (Aliases: num_boost_round, n_estimators, num_trees)
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'early_stopping_rounds': 50,
            # Try setting leaf_estimation_iterations to 1 or 5 to speed up the training on datasets with a small number of features.
            'leaf_estimation_iterations': trial.suggest_categorical('leaf_estimation_iterations', [1, 5]),
        },
        fold_count=3,
        partition_random_seed=RANDOM_SEED,
        verbose=False,
    )

    # return np.max(scores['test-AUC-mean'])
    return np.min(scores['test-Logloss-mean'])

const_params = {
    'task_type': 'GPU',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss', 
    'custom_metric': ['Logloss'],
    'random_seed': RANDOM_SEED,
    # fixed after tuning
    'random_strength': 0,

    # --------------- Speed Up Training -------------
    'bootstrap_type': 'Bayesian',
    # Ordered — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
    # Plain — The classic gradient boosting scheme.
    'boosting_type': 'Plain',
    # Try to set border_count of this parameter to 32 if training is performed on GPU. 
    # In many cases, this does not affect the quality of the model but significantly speeds up the training.
    'border_count': 32,
}