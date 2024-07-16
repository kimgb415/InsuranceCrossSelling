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
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 0.1, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_int('random_strength', 0, 15),
            'border_count': trial.suggest_int('border_count', 32, 255),
        },
        fold_count=5,
        partition_random_seed=RANDOM_SEED,
        verbose=False,
    )

    return np.max(scores['test-AUC-mean'])

const_params = {
    'task_type': 'GPU',
    'loss_function': 'CrossEntropy',
    'eval_metric': 'AUC', 
    'custom_metric': ['AUC'],
    'iterations': 100,
    'random_seed': RANDOM_SEED
}