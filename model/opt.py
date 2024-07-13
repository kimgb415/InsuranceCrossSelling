from keras import optimizers, callbacks
from .architecture import build_model
from hyperopt import hp, STATUS_OK
import numpy as np
from data.process import split_input_output, convert_to_tensorflow_dataset
import pandas as pd

def objective(params, train_df : pd.DataFrame, val_df : pd.DataFrame):
    train_dataset = convert_to_tensorflow_dataset(*(split_input_output(train_df)), int(params['batch_size']))
    val_dataset = convert_to_tensorflow_dataset(*(split_input_output(val_df)), int(params['batch_size']))

    features = len(train_df.columns) - 1
    output_dim = 1
    model = build_model(
        input_dimension=features, 
        dense_dimension=int(params['dense_dimension']), 
        output_dimension=output_dim, 
        block_depth=int(params['block_depth'])
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    history = model.fit(train_dataset,
                        epochs=1,
                        batch_size=int(params['batch_size']),
                        validation_data=val_dataset,
                        callbacks=[es],
                        verbose=0,
                        steps_per_epoch=500)
    
    val_loss = min(history.history['val_loss'])
    
    return {'loss': val_loss, 'status': STATUS_OK}

class SearchSpace():
    block_depth = np.arange(2, 5)
    dense_dimension = np.arange(50, 200, 10)
    learning_rate = [1e-4, 1e-3, 1e-2]
    batch_size = [32, 64, 128, 256]
    space = {}

    def __init__(self, block_depth = None, dense_dimension = None, learning_rate = None, batch_size = None):
        self.block_depth = block_depth if block_depth else self.block_depth
        self.dense_dimension = dense_dimension if dense_dimension else self.dense_dimension
        self.learning_rate = learning_rate if learning_rate else self.learning_rate
        self.batch_size = batch_size if batch_size else self.batch_size
        self.space = {
            'block_depth': hp.choice('block_depth', self.block_depth),
            'dense_dimension': hp.choice('dense_dimension', self.dense_dimension),
            'learning_rate': hp.choice('learning_rate', self.learning_rate),
            'batch_size': hp.choice('batch_size', self.batch_size)
        }

    def params_from_best(self, best):
        return {
            'block_depth': self.block_depth[best['block_depth']],
            'dense_dimension': self.dense_dimension[best['dense_dimension']],
            'learning_rate': self.learning_rate[best['learning_rate']],
            'batch_size': self.batch_size[best['batch_size']]
        }