from hyperopt import hp, STATUS_OK
import numpy as np
from data.process import split_input_output, convert_to_tensorflow_dataset
import pandas as pd
import xgboost as xgb
import optuna

RANDOM_STATE = 1
MAX_BIN_CHOICE = [2 ** i - 1 for i in [15, 18, 20, 22]]

xgb_tunable_hyperparams = {
    # learning_rate (Optional[float]) – Boosting learning rate (xgb’s “eta”)
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    # reg_lambda (Optional[float]) – L2 regularization term on weights (xgb’s lambda).
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(0.5)),
    # max_depth (Optional[int]) – Maximum tree depth for base learners.
    'max_depth': hp.uniformint('max_depth', 5, 20),
    # gamma (Optional[float]) – (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'gamma': hp.loguniform('gamma', np.log(0.000001), np.log(0.01)),
    # subsample (Optional[float]) – Subsample ratio of the training instance.
    'subsample': hp.uniform('subsample', 0.6, 1),
    # max_bin (Optional[int]) – If using histogram-based algorithm, maximum number of bins per feature
    "max_bin": hp.choice('max_bin', MAX_BIN_CHOICE),
    # 'min_child_weight': 1,
}



xgb_fixed_params = {
    # n_estimators (Optional[int]) – Number of gradient boosted trees. Equivalent to number of boosting rounds.
    'n_estimators': 500,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'device': 'cuda',
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': 30,
    # fixed after first case study
    'learning_rate': 0.05,
    'gamma': 0.001,
    'subsample': 0.8,
    'reg_lambda': 0.2,
    'max_bin': 32767,
    # fixed after second case study
    'max_depth': 27,
    'min_child_weight': 36,
    'enable_categorical': True,
}

def optuna_objective(trial : optuna.Trial, fixed_params, x_train, y_train, x_test, y_test):
    tunable_params = {
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.75),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 0.75),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 0.75),
    }
    fixed_params['random_state'] = trial.suggest_int('random_state', 1, 42)
    model = xgb.XGBClassifier(**tunable_params, **fixed_params)
    result = model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=250)
    best_validation_auc = max(result.evals_result()['validation_0']['auc'])

    return best_validation_auc

def objective(hyperparams, fixed_params, x_train, y_train, x_test, y_test):
    evals = [(x_test, y_test)]
    model = xgb.XGBClassifier(**hyperparams, **fixed_params)
    boosting = model.fit(x_train, y_train, eval_set=evals, verbose=100)

    best_validation_auc = max(boosting.evals_result()['validation_0']['auc'])

    return {'loss': -1 * best_validation_auc , 'status': STATUS_OK}


class XGBoostSearchSpace():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def Space(self):
        return {key: value for key, value in self.__dict__.items() if key != 'Space'}
