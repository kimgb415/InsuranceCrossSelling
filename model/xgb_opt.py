from hyperopt import hp, STATUS_OK
import numpy as np
from data.process import split_input_output, convert_to_tensorflow_dataset
import pandas as pd
import xgboost as xgb

xgb_tunable_hyperparams = {
    'n_estimators': 1000,
    'learning_rate': hp.loguniform('learning_rate', 0.001, 0.2),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'max_depth': hp.uniformint('max_depth', 3, 10),
    'gamma': hp.loguniform('gamma', -3, 1),
    # 'subsample': 0.5,
    # 'colsample_bytree': 1,
    # 'min_child_weight': 1,
    # 'max_bin': 256,
}



xgb_fixed_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'device': 'cuda',
}

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
