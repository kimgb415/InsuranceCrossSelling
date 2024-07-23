from lightgbm import LGBMClassifier
import optuna
import numpy as np
import pandas as pd

RANDOM_STATE = 1

# https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
lgmb_fixed_params = {
    'num_iterations': 500,
    'eval_metric': 'auc',
    'max_bin': 32767,
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': 50,
    'verbosity': -1,
    'num_threads': 12,
    # https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#gpu-setup
    # "device" : "gpu",

    # Use less data for faster tuning
    'bagging_freq': 5,
    'bagging_fraction': 0.75,
    }





def optuna_objective(trial : optuna.Trial, fixed_params, x_train : pd.DataFrame, y_train, x_test, y_test):
    tunable_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.75),
        'num_leaves': trial.suggest_int('num_leaves', 64, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
    }

    cat_features = list(x_train.select_dtypes(include=['object']).columns)

    model = LGBMClassifier(**tunable_params, **fixed_params)
    result = model.fit(x_train, y_train, eval_set=[(x_test, y_test)], categorical_feature=cat_features, eval_metric='auc')

    # _evals_result = {'valid_0': OrderedDict([('auc', [......]])])}
    best_validation_auc = max(result._evals_result['valid_0']['auc'])

    return best_validation_auc