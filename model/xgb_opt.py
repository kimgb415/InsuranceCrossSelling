from hyperopt import hp, STATUS_OK
import numpy as np
from data.process import split_input_output, convert_to_tensorflow_dataset
import pandas as pd
import xgboost as xgb

RANDOM_STATE = 1
MAX_BIN_CHOICE = [2 ** i - 1 for i in range(10, 20)]

xgb_tunable_hyperparams = {
    # learning_rate (Optional[float]) – Boosting learning rate (xgb’s “eta”)
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
    # reg_lambda (Optional[float]) – L2 regularization term on weights (xgb’s lambda).
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(0.5)),
    # max_depth (Optional[int]) – Maximum tree depth for base learners.
    'max_depth': hp.uniformint('max_depth', 5, 20),
    # gamma (Optional[float]) – (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'gamma': hp.loguniform('gamma', np.log(0.0001), np.log(0.1)),
    # subsample (Optional[float]) – Subsample ratio of the training instance.
    'subsample': hp.uniform('subsample', 0.6, 1),
    # max_bin (Optional[int]) – If using histogram-based algorithm, maximum number of bins per feature
    "max_bin": hp.choice('max_bin', MAX_BIN_CHOICE),
    # 'colsample_bytree': 1,
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
    'early_stopping_rounds': 50,
    # Referenced from https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516265
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
