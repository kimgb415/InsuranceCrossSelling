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
os.environ['CUDA_VISIBLE_DEVICES'] = f"1"

tabnet_params = {
    "optimizer_fn":torch.optim.Adam,
    "optimizer_params":dict(lr=2e-2),
    "scheduler_params":{
        "step_size":50, # how to use learning rate scheduler
        "gamma":0.9
    },
    "scheduler_fn":torch.optim.lr_scheduler.StepLR,
    # "mask_type":'entmax', # "sparsemax"
}

def tabnet_date_preprocess(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
    
    return df


def tabnet_inference(X_train: pd.DataFrame, X_test, y_train, y_test):
    # aug = ClassificationSMOTE(p=0.2)

    cat_idx = [i for i, col in enumerate(X_train.columns) if col in X_train.select_dtypes(include=['object']).columns]
    # https://github.com/dreamquark-ai/tabnet/issues/357#issuecomment-1030270567
    # train_test_split might not cover all unique categories in each categorical column
    X_merged = pd.concat([X_train, X_test])
    cat_dims = [len(X_merged[col].unique()) for col in cat_idx]
    emb_dims = [min(20, (x + 1) // 2) for x in cat_dims]
    
    model = TabNetClassifier(
        **tabnet_params,
        cat_idxs=cat_idx,
        cat_dims=cat_dims,
        cat_emb_dim=emb_dims,
    )

    model.fit(
        X_train=X_train.values, y_train=y_train.values, 
        eval_set=[(X_test.values, y_test.values)], eval_name=['valid'], eval_metric=['auc'], 
        max_epochs=2, patience=50, 
        batch_size=1024, virtual_batch_size=128, 
        num_workers=0, 
        drop_last=False,
    )

    plt.plot(model.history['valid_auc'])

    pass

def main(args):
    X_train, X_test, y_train, y_test = generate_train_test_data()
    X_train = tabnet_date_preprocess(X_train)
    X_test = tabnet_date_preprocess(X_test)
    if args.tune:
        pass
    elif args.inference:
        # convert pandas to numpy
        tabnet_inference(X_train, X_test, y_train, y_test)
        pass


if __name__ == "__main__":
    main(get_parser().parse_args())