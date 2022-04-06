import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def identity_block(x, kernel_size, filters, stage, block):
    """
    Block with no conv layer at shortcut.
    
    Arguments:
    x: input tensor.
    Kernel_size: kernel size of middle conv layer at main path
    filters: list of integers, the filterss of 3 conv layer at main path.
    stage: integer, current stage label, used for generating layer names.
    block: string/character, current block label, used for generating layer names.
    
    Returns:
    x: output tensor for the block.
    """
    
    # layer names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # filters tp be used in each layer of the block
    filters1, filters2, filters3 = filters
    
    # define shortcut
    x_shortcut = x
    
    # Main path
    x = Conv1D(filters = filters1, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2a')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)   
    
    x = Conv1D(filters = filters2, kernel_size = kernel_size, strides = 1, padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters = filters3, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2c')(x)

    # Add shortcut and main path
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

def conv_block(x, kernel_size, filters, stage, block, strides = 2):
    """
    Block with a conv1D layer at shortcut
    
    Arguments:
    x: input tensor.
    kernel_size: integer, kernel size of middle conv layer at main path.
    filters: list of integers, the filters of the conv1d layerr at main path.
    stage: integer, used to name the layers, depending on their position in the network
    block: 'a','b'..., current block label, used for generating layer names.
    
    Returns:
    x: Output tensor for the block.
    """
    
    # layers names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # filters
    filters1, filters2, filters3 = filters
    
    # Input to the shortcut
    x_shortcut = x

    # Main path
    x= Conv1D(filters1, 1, strides = strides, name = conv_name_base + '2a')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters2, kernel_size=kernel_size, padding = "same", name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, 1, name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = 2, name = bn_name_base + '2c')(x)
    
    # shortcut path
    x_shortcut = Conv1D(filters3, 1, strides = strides, name = conv_name_base + '1')(x_shortcut)
    x_shortcut = BatchNormalization(axis = 2, name = bn_name_base + '1')(x_shortcut)

    # Add shortcut and main path
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

def get_ResNet50_1D(input_shape = (5120, 1), kernel_size_stage1=3, kernel_size_blocks=3, n_out=1):
    """
    Implementation of ResNet50 for 1D data and regression.
    Modified from https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    Arguments:
    input_shape: shape of the 1D data.
    kernel_size_stage1: kernel size for the conv1d layer of satge 1
    kernel_size_blocks: kernel size for the 
    n_out:

    Returns:
    model: a Keras model instances. 
    """
    
    x_input = tf.keras.Input(input_shape)

    x = ZeroPadding1D(3)(x_input)
    
    # Stage 1
    x = Conv1D(64, kernel_size_stage1, strides=2, name = 'conv1')(x)
    x = BatchNormalization(axis = 2, name = 'bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    # Stage 2
    x = conv_block(x, kernel_size_blocks, filters = [16, 16, 64], stage = 2, block='a')
    x = identity_block(x, kernel_size_blocks, [16, 16, 64], stage=2, block='b')
    x = identity_block(x, kernel_size_blocks, [16, 16, 64], stage=2, block='c')

    # Stage 3
    x = conv_block(x, kernel_size_blocks, filters = [32,32,128], stage = 3, block='a')
    x = identity_block(x, kernel_size_blocks, [32,32,128], stage=3, block='b')
    x = identity_block(x, kernel_size_blocks, [32,32,128], stage=3, block='c')
    x = identity_block(x, kernel_size_blocks, [32,32,128], stage=3, block='d')

    # Stage 4
    x = conv_block(x, kernel_size_blocks, filters = [64, 64, 256], stage = 4, block='a')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=4, block='b')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=4, block='c')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=4, block='d')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=4, block='e')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=4, block='f')

    # Stage 5
    x = conv_block(x, kernel_size_blocks, filters = [64, 64, 256], stage = 5, block='a')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=5, block='b')
    x = identity_block(x, kernel_size_blocks, [64, 64, 256], stage=5, block='c')

    x = AveragePooling1D(5, name='avg_pool')(x)
    
    # flatten the output of the conv layers
    x = Flatten()(x)
    
    # Output fully connected layer
    x = Dense(n_out, name='fc-output')(x)
    
    model = Model(inputs = x_input, outputs = x, name='ResNet50_1D')

    return model
