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
    'num_threads': 24,
    'max_bin': MAX_BIN_CHOICE[1],
    # https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#gpu-setup
    # "device" : "gpu",

    # Fixed after first tuning
    'colsample_bytree': 0.25,
    'learning_rate': 0.1,

    # Fixed after second tuning
    'reg_lambda': 0.20811008331039144,

    # Fixed after third tuning
    'max_depth' : 32,

    # Use less data for faster tuning
    'bagging_freq': 5,
    'bagging_fraction': 0.75,
}

def optuna_objective(trial : optuna.Trial, fixed_params, x_train : pd.DataFrame, y_train, x_test, y_test):
    tunable_params = {
        # max num_leaves is 2^7=131072
        'num_leaves': trial.suggest_categorical('num_leaves', [2 ** i for i in range(5,  16)]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 200),
    }

    cat_features = list(x_train.select_dtypes(include=['object']).columns)

    model = LGBMClassifier(**tunable_params, **fixed_params)
    result = model.fit(x_train, y_train, eval_set=[(x_test, y_test)], categorical_feature=cat_features, eval_metric='auc')

    # _evals_result = {'valid_0': OrderedDict([('auc', [......]])])}
    best_validation_auc = max(result._evals_result['valid_0']['auc'])

    return best_validation_auc