import keras
from keras import layers

def fully_connected_block(x, dense_dimension, dropout_rate = 0.1):
    x = layers.Dense(dense_dimension, activation='relu')(x)  # (batch_size, dense_dimension)
    x = layers.Dropout(dropout_rate)(x)  # (batch_size, dense_dimension)
    x = layers.Dense(dense_dimension)(x)  # (batch_size, dense_dimension)
    x = layers.Dropout(dropout_rate)(x)  # (batch_size, dense_dimension)
    x = layers.BatchNormalization()(x)  # (batch_size, dense_dimension)
    
    return x

def residual_block(x, dense_dimension):
    x_shortcut = x  # (batch_size, dense_dimension)
    x = fully_connected_block(x, dense_dimension)  # (batch_size, dense_dimension)
    x = keras.layers.Add()([x, x_shortcut])  # (batch_size, dense_dimension)
    
    return x

def output_layer(x, dense_dimension, output_dimension=1):
    x = layers.Dense(dense_dimension, activation='relu')(x)  # (batch_size, dense_dimension)
    x = layers.Dense(output_dimension)(x)  # (batch_size, output_dimension)
    
    return x

def build_model(input_dimension, dense_dimension, output_dimension):
    input_layer = layers.Input(shape=(input_dimension,))  # (batch_size, input_dimension)
    x = input_layer  # (batch_size, input_dimension)
    x = fully_connected_block(x, dense_dimension)  # (batch_size, dense_dimension)
    
    for _ in range(5):
        x = residual_block(x, dense_dimension)  # (batch_size, dense_dimension)
    
    x = output_layer(x, dense_dimension, output_dimension)  # (batch_size, output_dimension)
    
    model = keras.models.Model(inputs=input_layer, outputs=x)
    
    return model
