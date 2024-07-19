import pandas as pd
import tensorflow as tf
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent

def read_csv_and_preprocess(file_path: Path):
    data = pd.read_csv(DATA_DIR / file_path)

    data = data.dropna()

    # convert object dtype to numerical
    data['Vehicle_Age'] = data['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'No': 0, 'Yes': 1})

    return data


# preprocessing so that catboost can recognize them as categorical features
def float64_to_int64(df: pd.DataFrame):
    # convert all numerical features into integers
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype('int64')

def retrieve_train_dev_test_for_catboost():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')

    dev = train[ :int(0.01 * len(train))]

    # # oversample the minority class
    # train_extra = pd.read_csv(DATA_DIR / 'train_extra.csv')
    # oversampled = train_extra[train_extra['Response'] == 1]

    # train = pd.concat([train[int(0.01 * len(train)): ], oversampled], ignore_index=True)
    train = train[int(0.01 * len(train)): ]

    # drop index column
    train = train.drop(columns=['id'])
    dev = dev.drop(columns=['id'])
    test = test.drop(columns=['id'])


    # convert float64 to int64
    float64_to_int64(train)
    float64_to_int64(dev)
    float64_to_int64(test)

    return train, dev, test


def retrieve_train_dev_test_as_category_for_xgboost():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')

    dev = train[ :int(0.01 * len(train))]
    train = train[int(0.01 * len(train)): ]

    # drop index column
    train = train.drop(columns=['id'])
    dev = dev.drop(columns=['id'])
    test = test.drop(columns=['id'])

    for col in train.columns:
        if col not in  ['Vehicle_Age', 'Gender', 'Vehicle_Damage']:
            continue
        train[col] = train[col].astype('category')
        dev[col] = dev[col].astype('category')
        test[col] = test[col].astype('category')
    
    return train, dev, test


def retrieve_train_dev_test_dataframe():
    train = read_csv_and_preprocess(DATA_DIR / 'train.csv')
    test = read_csv_and_preprocess(DATA_DIR / 'test.csv')

    # split train.csv it into train and dev
    dev = train[:int(0.01 * len(train))]
    train = train[int(0.01 * len(train)):]

    return train, dev, test


def split_input_output(df: pd.DataFrame):
    x = df.drop(columns=['Response'])
    y = df['Response']

    return x, y

# convert to tensorflow dataset
def convert_to_tensorflow_dataset(x : pd.DataFrame, y : pd.DataFrame, batch_size = 64):
    dataset = tf.data.Dataset.from_tensor_slices((x.values, y.values))
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)

    return dataset