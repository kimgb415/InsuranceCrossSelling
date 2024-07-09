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