# inception_model.py
# Functions and data used to build variants of the Inception and InceptionTime
# neural network models.
#
# References: 
# 1. "Going Deeper with Convolutions" https://arxiv.org/abs/1409.4842 
# 2. "InceptionTime: Finding AlexNet for Time Series Classification" https://arxiv.org/pdf/1909.04939

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, AveragePooling1D, BatchNormalization, \
    Concatenate, Conv1D, Cropping1D, Dense, Dropout, GlobalAveragePooling1D, MaxPool1D, \
    ReLU, ZeroPadding1D


# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------
layer_number = 1  # Global layer used for layer naming


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def ict_layer_number():
    """
    Return the next layer number as a string.
    Used for layer naming.
    """
    global layer_number
    lns = str(layer_number)
    layer_number += 1
    return lns

# ----------------------------------------------------------------------
def ict_bn_block(layer_in, cln, lns, rate=0.2):
    """
    Create and return a BatchNormalization + ReLU + Dropout sub-block.
    layer_in = The input layer to this sub-block
    cln = Parallel convolution layer number, used for naming
    lns = Inception block layer number string
    rate = Dropout rate (default = 0.2), if zero then no dropout layer is created
    """
    sfx = '{}_{}'.format(cln, lns)
    layer = BatchNormalization(name='BN'+sfx)(layer_in)
    layer = ReLU(name='ReLU'+sfx)(layer)
    if (rate > 0.0):
        layer = Dropout(rate=rate, name='Drop'+sfx)(layer)
    return layer

# ----------------------------------------------------------------------
def ict_fc_block(layer_in, nunits, act='relu', rate=0.2, lns=None):
    """
    Create and return a fully connected (dense) layer with an activation and dropout.
    layer_in = The input layer to this block
    nunits = Number of units
    act = Activation function to use (default = 'ReLU')
    rate = Dropout rate (default = 0.2), if zero then no dropout layer is created
    lns = Optional layer number string (default = use next layer number)
    """
    if lns is None:
        lns = ict_layer_number()
    layer = Dense(
        nunits, 
        activation=act, 
        name='FC_'+lns)(layer_in)
    if (rate > 0.0):
        layer = Dropout(rate=rate, name='Drop_'+lns)(layer)
    return layer
    
# ----------------------------------------------------------------------
def ict_inception_block(layer_in, 
    nconv=3, 
    nfilt=32, 
    ksize_start=16, 
    strides=1,
    conv_init='glorot_uniform',
    pool_size=8, 
    rate=0.2, 
    lns=None):
    """
    Create and return a single inception block.
    layer_in = The input layer to this block
    nconv = The number of parallel convolution layers in this block
    nfilt = The number of filters per convolution layer
    ksize_start = The starting kernel size; doubled for each additional parallel convolution layer
    strides = The convolution stride length
    conv_init = The convolution kernel initializer
    pool_size = The MaxPool window size
    rate = Dropout rate for the batch normalization sub-block
    lns = Optional layer number string (default = use next layer number)
    """
    conv_out = []
    concat_in = []
    
    if lns is None:
        lns = ict_layer_number()
    
    # Add zero padding to temporal dimension if needed to make an even number.
    if ((layer_in.shape[1] % 2) == 1):
        layer_in = ZeroPadding1D(padding=(0,1), name='Pad_'+lns)(layer_in)
    
    # Parallel convolution layers.
    ksize = ksize_start
    for i in range(nconv):
        sfx = '{}_{}'.format(i+1, lns)
        conv_out.append(Conv1D(
            filters=nfilt, 
            kernel_size=ksize, 
            strides=strides, 
            padding='same',
            kernel_initializer=conv_init,
            name='Conv'+sfx)(layer_in))
        concat_in.append(ict_bn_block(conv_out[i], i+1, lns, rate=rate))
        ksize *= 2

    # MaxPool skip layer.
    skip = MaxPool1D(pool_size=pool_size, strides=1, padding='same', name='MaxPool_'+lns)(layer_in)
    skip = Conv1D(nfilt, 1, strides=strides, padding='valid', kernel_initializer=conv_init, 
        name='Conv_skip_'+lns)(skip)
    concat_in.append(skip)
    
    # Concatenate the parallel layers.
    layer_out = Concatenate(axis=-1, name='Concat_'+lns)(concat_in)
    return layer_out

# ----------------------------------------------------------------------
def ict_model(model_input, **params):
    """
    Create and return an Inception Time model.
    model_input = The input layer to this model
    params:
        model_name = The name of the model to create
        classes = The number of arrhythmia detection classes to output
        depth = The number of inception blocks to create
        nconv = The number of parallel convolution layers per inception block
        nfilt = The number of filters per convolution
        ksize_start = The starting kernel size; doubled for each additional parallel convolution layer
        conv_init = The convolution kernel initializer
        pool_size = The MaxPool window size
        rate = Dropout rate
    """
    global layer_number
    layer_number = 1
    
    # Model parameters.
    model_name = params['model_name']
    classes = int(params['classes'])
    depth = int(params['depth'])
    nconv = int(params['nconv'])
    nfilt = int(params['nfilt'])
    ksize_start = int(params['ksize_start'])
    conv_init = params['conv_init']
    pool_size = int(params['pool_size'])
    rate = float(params['rate'])
    
    # Input block.
    layer = ict_inception_block(
        model_input, 
        nconv=nconv, 
        nfilt=nfilt, 
        ksize_start=ksize_start, 
        strides=1, 
        conv_init=conv_init,
        pool_size=pool_size, 
        rate=rate)
    
    # Additional inception blocks.
    for i in range(depth-1):
        layer = ict_inception_block(
            layer, 
            nconv=nconv, 
            nfilt=nfilt, 
            ksize_start=ksize_start, 
            strides=2,
            conv_init=conv_init,
            pool_size=pool_size, 
            rate=rate)
    
    # Fully connected layers.
    layer = GlobalAveragePooling1D(name='AvgPool')(layer)
    layer = ict_fc_block(layer, 128, rate=rate)
    layer = ict_fc_block(layer, 64, rate=min(rate, 0.1))
    layer = ict_fc_block(layer, 16, rate=0.0)
    lns = ict_layer_number()
    output = Dense(classes, activation='softmax', name='FC_'+lns)(layer)
    model = Model([model_input], output, name=model_name)
    return model
    