
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
from data.process import *
from functools import partial
from sklearn.linear_model import LogisticRegression
import pandas as pd

SEED = 1


def load_pretrained_model():
    xgboost_model = xgb.Booster(model_file='xgboost_model_new.json')
    lgbm_model = lgbm.Booster(model_file='lgbm_model_save')
    catboost_model = cb.CatBoostClassifier()
    catboost_model.load_model('catboost_model_categorical.cbm')

    return xgboost_model, lgbm_model, catboost_model

def save_xgboost_prediction(xgboost, x_train, x_valid):
    print("XGBoost predicting for training data")
    xgboost_pred = xgboost.predict(xgb.DMatrix(x_train, enable_categorical=True), iteration_range=(0, xgboost.best_iteration + 1))
    print("XGBoost predicting for validation data")
    xgboost_pred_valid = xgboost.predict(xgb.DMatrix(x_valid, enable_categorical=True), iteration_range=(0, xgboost.best_iteration + 1))

    np.save('xgboost_train_pred.npy', xgboost_pred)
    np.save('xgboost_valid_pred.npy', xgboost_pred_valid)


def save_lgbm_prediction(lgbm, x_train, x_valid):
    print("LGBM predicting for training data")
    lgbm_pred = lgbm.predict(x_train, num_iteration=lgbm.best_iteration)
    print("LGBM predicting for validation data")
    lgbm_pred_valid = lgbm.predict(x_valid, num_iteration=lgbm.best_iteration)

    np.save('lgbm_train_pred.npy', lgbm_pred)
    np.save('lgbm_valid_pred.npy', lgbm_pred_valid)

def save_catboost_prediction(catboost, x_train, x_valid):
    print("CatBoost predicting for training data")
    catboost_pred = catboost.predict_proba(x_train, ntree_end=catboost.best_iteration_)[:, 1]
    print("CatBoost predicting for validation data")
    catboost_pred_valid = catboost.predict_proba(x_valid, ntree_end=catboost.best_iteration_)[:, 1]

    np.save('catboost_train_pred.npy', catboost_pred)
    np.save('catboost_valid_pred.npy', catboost_pred_valid)


def load_train_valid_for_all():
    train_xgb, valid_xgb, _ = retrieve_train_dev_test_as_category_for_xgboost()
    train_cb, valid_cb, _ = retrieve_train_dev_test_for_catboost()
    
    x_train_xgb, y_train = split_input_output(train_xgb)
    x_train_cb, _ = split_input_output(train_cb)
    x_valid_xgb, y_valid = split_input_output(valid_xgb)
    x_valid_cb, _ = split_input_output(valid_cb)

    return x_train_xgb, x_train_cb, x_valid_xgb, x_valid_cb, y_train, y_valid

def stack_prediction():
    xgboost_pred = np.load('xgboost_train_pred.npy')
    xgboost_pred_valid = np.load('xgboost_valid_pred.npy')
    lgbm_pred = np.load('lgbm_train_pred.npy')
    lgbm_pred_valid = np.load('lgbm_valid_pred.npy')
    catboost_pred = np.load('catboost_train_pred.npy')
    catboost_pred_valid = np.load('catboost_valid_pred.npy')

    return (
        np.column_stack([xgboost_pred, lgbm_pred, catboost_pred]),
        np.column_stack([xgboost_pred_valid, lgbm_pred_valid, catboost_pred_valid]),
    )


def train_meta_model(train_predictions, valid_predictions, y_train, y_valid):
    model = cb.CatBoostClassifier(n_estimators=500, eval_metric='AUC', random_state=SEED)
    print("Fitting the meta model")
    model.fit(train_predictions, y_train, eval_set=(valid_predictions, y_valid), verbose=200)


    return model


def objective(trial: optuna.Trial, train_predictions, valid_predictions, y_train, y_valid):
    dataset = cb.Pool(
        data=train_predictions,
        label=y_train,
    )

    const_params = {
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC', 
        'custom_metric': ['AUC'],
        'random_seed': SEED,
        'use_best_model': True,
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

    params = {
        **const_params,
        'n_estimators': 600,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.15),
        # Maximum tree depth is 16
        'depth': trial.suggest_int('depth', 6, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 0.1, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.75, 1),
        # for imbalanced datasets
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        'early_stopping_rounds': 100,
        # Try setting leaf_estimation_iterations to 1 or 5 to speed up the training on datasets with a small number of features.
        'leaf_estimation_iterations': trial.suggest_categorical('leaf_estimation_iterations', [1, 5]),
    }

    model = cb.CatBoostClassifier(**params)
    model.fit(dataset, verbose=300, eval_set=(valid_predictions, y_valid))

    return model.best_score_['validation']['AUC']