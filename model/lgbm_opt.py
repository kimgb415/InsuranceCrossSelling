from lightgbm import LGBMClassifier
import optuna
import numpy as np
import pandas as pd

RANDOM_STATE = 1
MAX_BIN_CHOICE = [2 ** i - 1 for i in [15, 18, 20, 22]]

# https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
lgbm_fixed_params = {
    'n_estimators': 500,
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': 50,
    'verbosity': -1,
    'num_threads': 12,
    # https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#gpu-setup
    # "device" : "gpu",

    # Fixed after first tuning
    'colsample_bytree': 0.25,
    'max_depth': 12,
    'num_leaves': 220,

    # Use less data for faster tuning
    'bagging_freq': 5,
    'bagging_fraction': 0.75,
}

def optuna_objective(trial : optuna.Trial, fixed_params, x_train : pd.DataFrame, y_train, x_test, y_test):
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    tunable_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.5),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
        'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, scale_pos_weight]),
        'max_bin': trial.suggest_categorical('max_bin', MAX_BIN_CHOICE)
    }

    cat_features = list(x_train.select_dtypes(include=['object']).columns)

    model = LGBMClassifier(**tunable_params, **fixed_params)
    result = model.fit(x_train, y_train, eval_set=[(x_test, y_test)], categorical_feature=cat_features, eval_metric='auc')

    # _evals_result = {'valid_0': OrderedDict([('auc', [......]])])}
    best_validation_auc = max(result._evals_result['valid_0']['auc'])

    return best_validation_auc