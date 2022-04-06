from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers


def get_ConvNet1D(n_points):
    
    model = Sequential()
    model.add(Conv1D(
        filters=32,
        kernel_size=9,
        kernel_regularizer=l2(0.0001),
        padding="same",
        activation='relu',
        input_shape=(n_points, 1),
        name='conv1d_1'
    ))
    model.add(MaxPooling1D(
        pool_size=2,
        strides=2,
        name='maxPool_1'
    ))
    model.add(Conv1D(
        filters=64,
        kernel_size=9,
        kernel_regularizer=l2(0.0001),
        padding="same",
        activation='relu',
        name='conv1d_2'
    ))
    model.add(MaxPooling1D(
        pool_size=2,
        strides=2,
        name='maxPool_2'
    ))
    model.add(Conv1D(
        filters=128,
        kernel_size=9,
        kernel_regularizer=l2(0.0001),
        padding="same",
        activation='relu',
        name='conv1d_3'
    ))
    model.add(MaxPooling1D(
        pool_size=2,
        strides=2,
        name='maxPool_3'
    ))
    model.add(Flatten())
    model.add(Dense(
        128,
        kernel_regularizer=l2(0.0001),
        name='Dense'
    ))
    model.add(Dense(1))

    return(model)
