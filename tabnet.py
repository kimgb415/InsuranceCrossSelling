# https://github.com/dreamquark-ai/tabnet/blob/develop/census_example.ipynb
from sklearn_gbm import generate_train_test_data
from sklearn_gbm import get_parser
from pytorch_tabnet.augmentations import ClassificationSMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import os
import optuna
from pprint import pprint


SEED = 1

tabnet_tuning_params = {
    "optimizer_fn":torch.optim.Adam,
    "optimizer_params":dict(lr=2e-2),
    "scheduler_params":{
        "step_size":50, # how to use learning rate scheduler
        "gamma":0.9
    },
    "scheduler_fn":torch.optim.lr_scheduler.StepLR,
    'device_name':'cuda',
    'seed':SEED,
    
    # tuned
    'gamma': 1.0128779115960516,
    'n_independent': 4,
    'n_shared': 5,
}


def tabnet_data_preprocess(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
    
    return df


def objective(trial : optuna.Trial, X_train, y_train, X_test, y_test):
    cat_idx = [i for i, col in enumerate(X_train.columns) if col in X_train.select_dtypes(include=['object']).columns]
    # https://github.com/dreamquark-ai/tabnet/issues/357#issuecomment-1030270567
    # train_test_split might not cover all unique categories in each categorical column
    X_merged = pd.concat([X_train, X_test])
    cat_dims = [len(X_merged[col].unique()) for col in cat_idx]
    emb_dims = [min(20, (x + 1) // 2) for x in cat_dims]

    params = {
        **tabnet_tuning_params,
        'n_d': trial.suggest_int('n_b', 20, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.5, 3.0),
        'momentum': trial.suggest_float('momentum', 0.01, 0.4),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-7, 1e-3),
    }

    # According to the paper n_d=n_a is usually a good choice.
    params['n_a'] = params['n_d']

    model = TabNetClassifier(
        **params,
        cat_idxs=cat_idx,
        cat_dims=cat_dims,
        cat_emb_dim=emb_dims,
    )

    batch_size = trial.suggest_categorical('batch_size', [4096, 8192, 16384, 32768])
    virutal_batch_size = batch_size // 4

    model.fit(
        X_train=X_train.values, y_train=y_train.values, 
        eval_set=[(X_test.values, y_test.values)], eval_name=['valid'], eval_metric=['auc'], 
        max_epochs=1, 
        batch_size=batch_size,
        virtual_batch_size=virutal_batch_size,
        drop_last=False,
        # feature importance computing is extremely slow
        compute_importance=False,
    )

    return max(model.history['valid_auc'])


def tabnet_training(X_train: pd.DataFrame, X_test, y_train, y_test):
    # aug = ClassificationSMOTE(p=0.2)

    cat_idx = [i for i, col in enumerate(X_train.columns) if col in X_train.select_dtypes(include=['object']).columns]
    # https://github.com/dreamquark-ai/tabnet/issues/357#issuecomment-1030270567
    # train_test_split might not cover all unique categories in each categorical column
    X_merged = pd.concat([X_train, X_test])
    cat_dims = [len(X_merged[col].unique()) for col in cat_idx]
    emb_dims = [min(20, (x + 1) // 2) for x in cat_dims]

    params = {
        "optimizer_fn":torch.optim.Adam,
        "optimizer_params":dict(lr=0.1),
        "scheduler_params":{
            "step_size": 5,
            "gamma":0.5
        },
        "scheduler_fn":torch.optim.lr_scheduler.StepLR,
        "mask_type":'entmax',
        'device_name':'cuda',
        'seed':SEED,

        # Tuned
        'n_d':64,
        'n_a':64,
        'n_steps': 5,
        'gamma': 1.0128779115960516,
        'n_independent': 4,
        'n_shared': 5,
        'lambda_sparse': 0.0001,
    }

    
    model = TabNetClassifier(
        **params,
        cat_idxs=cat_idx,
        cat_dims=cat_dims,
        cat_emb_dim=emb_dims,
    )

    model.fit(
        X_train=X_train.values, y_train=y_train.values, 
        eval_set=[(X_test.values, y_test.values)], eval_name=['valid'], eval_metric=['auc'], 
        max_epochs=25, patience=10, 
        batch_size=16384, virtual_batch_size=1024, 
        num_workers=0, 
        drop_last=False,
        compute_importance=False,
    )
    model.save_model('tabnet_model')

    pass


def main(args):
    X_train, X_test, y_train, y_test = generate_train_test_data()
    X_train = tabnet_data_preprocess(X_train)
    X_test = tabnet_data_preprocess(X_test)
    if args.tune:
        pass
    elif args.training:
        # convert pandas to numpy
        tabnet_training(X_train, X_test, y_train, y_test)
        pass


if __name__ == "__main__":
    main(get_parser().parse_args())