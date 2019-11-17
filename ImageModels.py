import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, SpatialDropout2D, \
        Activation, Input, merge, Add, Concatenate, Multiply
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
from keras.layers.core import Lambda
import keras.layers

from keras.optimizers import Adam, SGD
from keras import backend as K
import keras.callbacks
import keras.initializers
import keras.regularizers

# from alexnet_utils import preprocess_image_batch
from alexnet_additional_layers import split_tensor, cross_channel_normalization
# from decode_predictions import decode_classnames_json, decode_classnumber

from MnistModel2 import *

# AlexNet, a bit modified for existing train images
def MyAlexnetModel(classCount=25):
    """
    Returns a keras model for AlexNet, achieving roughly 80% at ImageNet2012 validation set

    Model and weights from
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    and only slightly modified to work with TF backend
    """

    from keras.utils.layer_utils import convert_all_kernels_in_model

    if 0:
        # K.set_image_dim_ordering('th')
        K.set_image_data_format('channels_first')
        inputs = Input(shape=(3, 227, 227))
        chanAxis = 1
    else:
        inputs = Input(shape=(227, 227, 3))
        chanAxis = 3

    conv_1 = Conv2D(96, 11, strides=4, activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
                # 11  4  4 = 19, stride 8
    conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    new_convs =[Conv2D(128, 5, activation="relu", name='conv_2_' + str(i + 1))
        (split_tensor(axis=chanAxis, ratio_split=2, id_split=i)(conv_2)
         ) for i in range(2)]
                # 2 * 8 <- (shifted)    19 8 8 8 8 = 51
    conv_2 =  Concatenate(axis=chanAxis, name="conv_2")(new_convs)
    # conv_2 = merge([ \
    #     Conv2D(128, 5, activation="relu", name='conv_2_' + str(i + 1))
    #     (split_tensor(ratio_split=2, id_split=i)(conv_2)
    #      ) for i in range(2)])  # , mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
                # 51 8 8 = 67, stride 16
    conv_3 = cross_channel_normalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, activation='relu', name='conv_3')(conv_3)
                # 2 * 8 + 16 <-    67 16 16

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Concatenate(axis=chanAxis, name="conv_4")([
        Conv2D(192, 3, activation="relu", name='conv_4_' + str(i + 1))(
            split_tensor(axis=chanAxis, ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)])   # , mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Concatenate(axis=chanAxis, name="conv_5")([
        Conv2D(128, 3, activation="relu", name='conv_5_' + str(i + 1))(
            split_tensor(axis=chanAxis, ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)])   # mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(25, name='dense_3')(dense_3)   # Class count
    prediction = Activation("softmax", name="softmax")(dense_3)

    m = Model(input=inputs, output=prediction)

    # if weights_path is None:
        # weights_path = 'Data/alexnet_weights.h5'
    # if not weights_path is None:
    #     m.load_weights(weights_path)
    # Model was trained using Theano backend
    # This changes convolutional kernels from TF to TH, great accuracy improvement
    convert_all_kernels_in_model(m)

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # m.compile(optimizer=sgd, loss='mse')

    return m

def CImageModel():
    import tensorflow as tf

    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    towerCount = 2
    mult = 4
    additLayerCounts = (0, 0)  # (2, 2)
    towerWeightsKerasVar = tf.compat.v1.Variable(np.ones([towerCount * 2 + 2]) * 5 / 9,
                                                 dtype=tf.float32, name='tower_weights')
    towerWeightsKerasVar._trainable = False    # "Doesn't help against Tensor.op is meaningless when eager execution is enabled."
    towerWeights = Input(shape=(towerCount, ), tensor=towerWeightsKerasVar)
        # Looks like input, should be declared in model's input, but (if the tensor parameter is present)
        # should not be supplied to the net (in model.fit and so on)
    # towerWeights = keras.layers.Layer(name='tower_weights')
    # towerWeights.add_weight(shape=[towerCount], trainable=False, initializer='ones')
    # towerWeights = Input(tensor=K.variable(np.ones([4, 1, 1]), name="ones_variable"))

    conv_1 = Conv2D(mult * 9, 5, strides=(2, 2), activation='relu',
                    name='conv_11')(inputs)
        # 12 * 12

    conv_1s = []     # Lists only for creating concatenated level for simple data extraction
    add_23s = []
    last_tower_convs = []
    for towerInd in range(towerCount):
        t_conv_2 = conv_1
        # t_conv_2 = BatchNormalization()(t_conv_2)
        t_conv_2 = split_tensor(axis=3, ratio_split=towerCount, id_split=towerInd)(t_conv_2)
        t_conv_2 = Conv2D(mult * 4, 3, strides=(1, 1),
                          activation='relu' if towerInd == 0 else 'sigmoid',
                          name='conv_12_%d' % towerInd)(t_conv_2)
            # 10 * 10

        if additLayerCounts[0] < 1:
            last_tower_conv = t_conv_2
            # With smallerInputAreas == False actually there are no towers here
        else:
            # t_conv_3 = BatchNormalization()(t_conv_2)
            # t_conv_3 = Dropout(0.3)(t_conv_2)
            t_conv_3 = cross_channel_normalization(name='cross_chan_norm_1%d' % towerInd)(t_conv_2)
            # t_conv_3 = ZeroPadding2D((1, 1))(t_conv_3)
            t_conv_3 = Conv2D(mult * 4, 3, padding='same', strides=(1, 1),
                              activation='relu' if towerInd == 0 else 'sigmoid', # activity_regularizer=keras.regularizers.l1(1e-6),
                              name='conv_13_%d' % towerInd,
                              kernel_initializer=MyInitializer)(t_conv_3)
                # Output - 4 * 4
            # conv_3s.append(t_conv_3)
            t_conv_3 = Add(name='add_123_%d' % towerInd)([t_conv_2, t_conv_3])
            add_23s.append(t_conv_3)
        # # t_conv_3 = MaxPooling2D((2, 2), strides=(2, 2))(t_conv_2)

            if additLayerCounts[0] < 2:
                last_tower_conv = t_conv_3
            else:
                t_conv_4 = BatchNormalization()(t_conv_3)
                # t_conv_4 = ZeroPadding2D((1, 1))(t_conv_4)
                t_conv_4 = Conv2D(mult * 4, 3, padding='same', strides=(1, 1),
                                  activation='relu' if towerInd == 0 else 'sigmoid', # activity_regularizer=keras.regularizers.l1(0.03),
                                  name='conv_14_%d' % towerInd,
                                  kernel_initializer=MyInitializer)(t_conv_4)
                    # Output - 4 * 4
                # conv_4s.append(t_conv_4)
                t_conv_4 = Multiply()([t_conv_4, get_tensor_array_element(towerInd * 2)(towerWeights)])
                    # Switches off level 14
                t_conv_4 = Add(name='add_134_%d' % towerInd)([t_conv_3, t_conv_4])
                # t_conv_4 = Multiply()([ReLU()(t_conv_4), get_tensor_array_element(towerInd)(towerWeights)])
                        # K.constant(value=np.ones([1, 1, 1]), dtype='float32')])   # if towerInd in [0, 3] else \
                            # K.constant(value=0, dtype='float32')])
                     # split_tensor(axis=1, ratio_split=towerCount, id_split=towerInd)(towerWeights)])
                last_tower_conv = t_conv_4
                # add_34s.append(t_conv_4)

        last_tower_conv = Multiply()([last_tower_conv, get_tensor_array_element(towerInd)(towerWeights)])
        last_tower_convs.append(MaxPooling2D((2, 2), strides=(2, 2))(last_tower_conv))
            # 5 * 5

    conv_2 = Concatenate(axis=3, name='conv_1_adds')(last_tower_convs)
        # This layer produces 8*1 gradients tensor, so special conv_2/3 was added
    conv_2 = cross_channel_normalization(name="cross_chan_norm_1u")(conv_2)
    conv_2 = Conv2D(mult * 12, 2, strides=(1, 1),
                       activation='relu', # activity_regularizer=keras.regularizers.l1(1e-6),
                       name='conv_21')(conv_2)
        # 4 * 4

    last_tower_convs = []
    for towerInd in range(towerCount):
        t_conv_2 = conv_2
        t_conv_2 = BatchNormalization()(t_conv_2)
        t_conv_2 = split_tensor(axis=3, ratio_split=towerCount, id_split=towerInd)(t_conv_2)
        t_conv_2 = Conv2D(mult * 8, 2, strides=(1, 1),
                          activation='relu' if towerInd == 0 else 'sigmoid',
                          name='conv_22_%d' % towerInd)(t_conv_2)
            # 3 * 3

        if additLayerCounts[1] < 1:
            last_tower_conv = t_conv_2
            # With smallerInputAreas == False actually there are no towers here
        else:
            # t_conv_3 = BatchNormalization()(t_conv_2)
            # t_conv_3 = Dropout(0.3)(t_conv_2)
            t_conv_3 = cross_channel_normalization(name='cross_chan_norm_2%d' % towerInd)(t_conv_2)
            # t_conv_3 = ZeroPadding2D((1, 1))(t_conv_3)
            t_conv_3 = Conv2D(mult * 8, 2, padding='same', strides=(1, 1),
                              activation='relu' if towerInd == 0 else 'sigmoid', # activity_regularizer=keras.regularizers.l1(1e-6),
                              name='conv_23_%d' % towerInd,
                              kernel_initializer=MyInitializer)(t_conv_3)
            # conv_3s.append(t_conv_3)
            t_conv_3 = Add(name='add_223_%d' % towerInd)([t_conv_2, t_conv_3])
            # add_23s.append(t_conv_3)
        # # t_conv_3 = MaxPooling2D((2, 2), strides=(2, 2))(t_conv_2)

            if additLayerCounts[1] < 2:
                last_tower_conv = t_conv_3
            else:
                t_conv_4 = BatchNormalization()(t_conv_3)
                # t_conv_4 = ZeroPadding2D((1, 1))(t_conv_4)
                t_conv_4 = Conv2D(mult * 8, 2, padding='same', strides=(1, 1),
                                  activation='relu' if towerInd == 0 else 'sigmoid', # activity_regularizer=keras.regularizers.l1(0.03),
                                  name='conv_24_%d' % towerInd,
                                  kernel_initializer=MyInitializer)(t_conv_4)
                    # Output - 4 * 4
                # conv_4s.append(t_conv_4)
                t_conv_4 = Multiply()([t_conv_4, get_tensor_array_element(towerInd * 2 + 1)(towerWeights)])
                    # Switches off level 14
                t_conv_4 = Add(name='add_234_%d' % towerInd)([t_conv_3, t_conv_4])
                # t_conv_4 = Multiply()([ReLU()(t_conv_4), get_tensor_array_element(towerInd)(towerWeights)])
                        # K.constant(value=np.ones([1, 1, 1]), dtype='float32')])   # if towerInd in [0, 3] else \
                            # K.constant(value=0, dtype='float32')])
                     # split_tensor(axis=1, ratio_split=towerCount, id_split=towerInd)(towerWeights)])
                last_tower_conv = t_conv_4
                # add_34s.append(t_conv_4)

        last_tower_conv = Multiply()([last_tower_conv, get_tensor_array_element(towerCount + towerInd)(towerWeights)])
        # last_tower_convs.append(MaxPooling2D((2, 2), strides=(2, 2))(last_tower_conv))
        last_tower_convs.append(last_tower_conv)

    conv_last = Concatenate(axis=3, name='conv_2_adds')(last_tower_convs)
    conv_last = cross_channel_normalization(name='cross_chan_norm_2u')(conv_last)
    # conv_2 = Conv2D(32, 3, strides=(1, 1), activation='relu', name='conv_2')(conv_1)
    # conv_2 = DepthwiseConv2D(3, depth_multiplier=4, activation='relu', name='conv_2')(conv_1)

    # conv_next = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
    # conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    # conv_2 = ZeroPadding2D((2, 2))(conv_2)
    # conv_next = Conv2D(20, 3, strides=1, activation='relu', name='conv_3')(conv_next)
    # conv_next = DepthwiseConv2D(3, depth_multiplier=2, activation='relu', name='conv_3')(conv_next)

    dense_1 = BatchNormalization()(conv_last)
    # dense_1 = Dropout(0.3)(conv_3)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(mult * 32, activation='relu', # activity_regularizer=keras.regularizers.l1(1e-6),
                    name='dense_1')(dense_1)

    # dense_2 = BatchNormalization()(dense_1)
    dense_2 = Dropout(0.4)(dense_1)
    dense_2 = Dense(mult * 12, name='dense_2')(dense_2)

    # dense_3 = BatchNormalization()(dense_2)
    dense_3 = Dropout(0.4)(dense_2)
    dense_3 = Dense(25, name='dense_3')(dense_3)     # Class count
    # y = K.variable(value=2.0)
    # meaner=Lambda(lambda x: K.mean(x, axis=1) )
    # p = Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)
    prediction = Activation("softmax", name="softmax")(pow(1.2)(dense_3))

    model = Model(inputs=[inputs, towerWeights], outputs=prediction)
    print(model.summary())

    # if not weights_path is None:
    #     model.load_weights(weights_path)

    # convert_all_kernels_in_model(model)

    # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(learning_rate=0.001, decay=1e-5)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    model.variables = {'towers_weights': towerWeightsKerasVar}
    model.towerCount = towerCount

    model.debug_layers = dict()
    for blockInd in range(2):
        name = 'conv_%d2' % (blockInd + 1)
        model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
        if additLayerCounts[blockInd] >= 1:
            name = 'conv_%d3' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d23' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
        if additLayerCounts[blockInd] >= 2:
            name = 'conv_%d4' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d34' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)

    return model

