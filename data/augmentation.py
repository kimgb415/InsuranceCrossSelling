from imblearn.over_sampling import SMOTE
from .process import split_input_output


RANDOM_SEED = 1


def oversample_train_data(train):
    smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED)
    x_train, y_train = split_input_output(train)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    
    return x_train, y_train