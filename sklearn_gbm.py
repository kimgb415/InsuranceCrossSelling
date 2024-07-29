import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser
import optuna
from functools import partial
from my_log import console_handler, file_handler
import logging

STORAGE_NAME = 'sqlite:///sklearn_hist.db'
STUDY_NAME = 'sklearn_hist_0'
LOG = logging.getLogger(__name__)


fixed_params = {
    'loss': 'log_loss',
    # 'learning_rate': 0.1,
    'random_state': 42,
    'scoring': 'accuracy',
    'max_iter': 500,
    'early_stopping': True,
    'n_iter_no_change': 50,
}

def generate_train_test_data():
    df = pd.read_csv('data/train.csv')

    X = df.drop(columns=['id', 'Response'])
    y = df['Response']

    # # Handle categorical features
    # # Identify the categorical columns
    # categorical_cols = X.select_dtypes(include=['object']).columns

    # # Encode the categorical columns using OrdinalEncoder
    # encoder = OrdinalEncoder()
    # X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_predict(X_train, X_test, y_train, y_test):
    # Initialize the HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        **fixed_params,
        max_depth=10,
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)


def tuning(trial: optuna.Trial, X_train: pd.DataFrame, X_test, y_train, y_test):
    LOG.info(f'Trial Number: {trial.number}')
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-5, 1e-1, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 255),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 50, 1000),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        # 'categorical_features': trial.suggest_categorical(
        #     'categorical_features', 
        #     [
        #         [i for i in range(X_train.shape[1])], 
        #         [col for col in X_train.columns.values if X_train[col].dtype == 'object']
        #     ]
        # ),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
    }

    clf = HistGradientBoostingClassifier(
        **fixed_params,
        **params,
        categorical_features=[col for col in X_train.columns.values if X_train[col].dtype == 'object']
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    LOG.info(f'Accuracy: {accuracy:.4f}')
    

    return accuracy


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--inference', action='store_true', help='Run inference')
    parser.add_argument('--training', action='store_true', help='Train from scratch')

    return parser

def main(args):
    if args.tune:
        X_train, X_test, y_train, y_test = generate_train_test_data()
        study = optuna.create_study(direction='maximize', study_name=STUDY_NAME, storage=STORAGE_NAME)
        objective = partial(tuning, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        study.optimize(objective, n_trials=40)
    elif args.inference:
        X_train, X_test, y_train, y_test = generate_train_test_data()
        train_predict(X_train, X_test, y_train, y_test)



if __name__ == '__main__':
    args = get_parser().parse_args()
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(console_handler)
    LOG.addHandler(file_handler)

    main(args)


