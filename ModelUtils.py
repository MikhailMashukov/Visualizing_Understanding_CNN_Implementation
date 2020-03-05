import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, SpatialDropout2D, \
        Activation, Input, merge, Add, Concatenate, Multiply
# from keras.layers.convolutional import Conv2D, DepthwiseConv2D, MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import ReLU, ELU
import keras.layers

from keras import backend as K
import keras.callbacks
import keras.initializers
import keras.regularizers

# from alexnet_utils import preprocess_image_batch

from keras.layers.core import Lambda
from alexnet_additional_layers import split_tensor, cross_channel_normalization
# from decode_predictions import decode_classnames_json, decode_classnumber


# def fractional_max_pool(x): # , ratio):
#     return tf.nn.fractional_max_pool(x, [1, ratio, ratio, 1], overlapping=True)[0]
# Actual ratio can be higher due to rounding to pixels
def fractional_max_pool(ratio, overlapping=False, **kwargs):
    def f(X):
        return tf.nn.fractional_max_pool(X, [1, ratio, ratio, 1], overlapping=overlapping)[0]

    return Lambda(f, **kwargs)

def cut_image_tensor(x0, xk, y0, yk, **kwargs):
    def f(X):
        output = X[:, y0 : yk, x0 : xk, :]
        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[1] = yk - y0
        output_shape[2] = xk - x0
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)

def pow(y, **kwargs):
    def f(X):
        output = K.pow(K.abs(X), y)
        return output

    def g(input_shape):
        return input_shape

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

g_excLayerCount = 0

# SE module performing inter-channel weighting.
def squeeze_excitation_layer(x, ratio, activation='elu'): # , out_dim):
    global g_excLayerCount

    squeeze = keras.layers.pooling.GlobalAveragePooling2D()(x)

    g_excLayerCount += 1
    out_dim = x.shape[3]   # Channels_last
    excitation = Dense(units=out_dim // ratio, name='dense_exc_%d' % (g_excLayerCount * 2 - 1))(squeeze)
    excitation = Activation(activation)(excitation)
    excitation = Dense(units=out_dim, name='dense_exc_%d' % (g_excLayerCount * 2))(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = keras.layers.Reshape((1, 1, out_dim))(excitation)

    scale = keras.layers.multiply([x, excitation])
    return scale

# Implementation of Squeeze-and-Excitation(SE) block as described in https://arxiv.org/abs/1709.01507.
# From https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py.
# There also double, CBAM blocks, there. They apply the same at first to channels and then - to pixels
def SeBlock(input_feature, ratio=8):
    global g_excLayerCount

    g_excLayerCount += 1
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = keras.layers.pooling.GlobalAveragePooling2D()(input_feature)
    se_feature = keras.layers.Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
					   activation='relu',
                       name='dense_exc_%d' % (g_excLayerCount * 2 - 1),
					   # kernel_initializer='he_normal',
                       kernel_initializer=My1PlusInitializer(1.0 / 64),
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
					   activation='sigmoid',
                       name='dense_exc_%d' % (g_excLayerCount * 2),
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = keras.layers.Permute((3, 1, 2))(se_feature)

    se_feature = keras.layers.multiply([input_feature, se_feature])
    return se_feature

def near1Regularizer(activationVector):
    return 0.01 * K.mean((1 - activationVector) ** 2)

def SeBlock4_PrevChannels(input_feature, prevChans, ratio=8):
    global g_excLayerCount

    g_excLayerCount += 1
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = keras.layers.pooling.GlobalAveragePooling2D()(input_feature)
    se_feature = keras.layers.Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    if not prevChans is None:
        se_feature = Concatenate(axis=3)([se_feature, prevChans])
    newChans = se_feature
    se_feature = Dense(channel // ratio,
					   activation='relu',
                       name='dense_exc_%d' % (g_excLayerCount * 2 - 1),
					   # kernel_initializer='he_normal',
                       kernel_initializer=My1PlusInitializer(1.0 / 256),
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       name='dense_exc_%d' % (g_excLayerCount * 2),
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros',
                       activity_regularizer=near1Regularizer)(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = keras.layers.Permute((3, 1, 2))(se_feature)

    se_feature = keras.layers.multiply([input_feature, se_feature])
    return (se_feature, newChans)


def get_tensor_array_element(x):
    def f(X):                               # Here we have e.g. np.ones([4])
        return X[x : x + 1]

    def g(input_shape):
        output_shape = list(input_shape)    # Here - e.g. (None, 4)
        output_shape[-1] = 1
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape))

def isMatchedLayer(layerName, layerToFindNames, allowCombinedLayers):
    for layerToFindName in layerToFindNames:
        if layerName == layerToFindName or \
           (allowCombinedLayers and layerName[ : len(layerToFindName) + 1] == layerToFindName + '_'):
            return True
    return False

def concatenateLayersByNameBegin(model, nameBegin):
    foundLayers = []
    l = len(nameBegin)
    for layer in model.layers:
        if layer.name[:l] == nameBegin:
            foundLayers.append(layer.get_output_at(0))   # output)

    if not foundLayers:
        return None
    elif len(foundLayers) == 1:
        return foundLayers[0]
    else:
        return Concatenate(axis=3, name=nameBegin)(foundLayers)


def MyInitializer(shape, dtype=None):
    v = keras.initializers.VarianceScaling(scale=1.0/16,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=None)
    return v(shape, dtype)

def MyVarianceScalingInitializer(coef=1.0/16):
    def MyInitializer(shape, dtype=None):
        v = keras.initializers.VarianceScaling(scale=coef,
                               mode='fan_avg',
                               distribution='uniform',
                               seed=None)
        return v(shape, dtype)
    return MyInitializer

def My1PlusInitializer(coef=1.0/16):
    def MyInitializer(shape, dtype=None):
        size = shape[-1]
        # print('My1PlusInitializer shape: ', shape)
        v = keras.initializers.RandomNormal(mean=1.0 / size, stddev=1.0 / size * coef,
                               seed=None)
        return v(shape, dtype)
    return MyInitializer
