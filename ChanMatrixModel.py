import numpy as np
import tensorflow as tf

if 1:
    from keras.models import Model
    from keras.layers import Flatten, Reshape, Dense, Dropout, SpatialDropout2D, \
            Activation, Input, merge, Add, Concatenate, Multiply
    from keras.layers.convolutional import Conv2D, Conv3D, DepthwiseConv2D,\
            MaxPooling2D, MaxPooling3D, ZeroPadding2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import ReLU
    from keras.layers.core import Lambda
    import keras.layers

    from keras.optimizers import Adam, SGD
    from keras import backend as K
    import keras.callbacks
    import keras.initializers
    import keras.regularizers
    from keras.utils import convert_all_kernels_in_model

from alexnet_additional_layers import split_tensor, cross_channel_normalization

import DeepOptions
from ModelUtils import *
# from ImageModels import ModelBlock

# Takes the same arguments as Conv2D and
# optional doubleSizeLayerNames
def ModelBlock2(filters, kernel_size, **kwargs):
    sizeMult = 1
    if 'doubleSizeLayerNames' in kwargs:
        if 'name' in kwargs and isMatchedLayer(kwargs['name'], kwargs['doubleSizeLayerNames'], True):
            print("Layer '%s' has double size" % kwargs['name'])
            sizeMult = 2
        del kwargs['doubleSizeLayerNames']

    if 1:
        return Conv2D(filters * sizeMult, kernel_size, **kwargs)
    else:
        def blockLayer(x):
            conv = Conv2D(filters * sizeMult, kernel_size,
                          **kwargs)(x)
            x = se_block(conv, 4)
            return x

        return blockLayer

def ModelBlock2_3D(filters, kernel_size, **kwargs):
    sizeMult = 1
    if 'doubleSizeLayerNames' in kwargs:
        if 'name' in kwargs and isMatchedLayer(kwargs['name'], kwargs['doubleSizeLayerNames'], True):
            print("Layer '%s' has double size" % kwargs['name'])
            sizeMult = 2
        del kwargs['doubleSizeLayerNames']

    return Conv3D(filters * sizeMult, kernel_size, **kwargs)

def ChanMatrixModel(classCount=DeepOptions.classCount, doubleSizeLayerNames=[]):
    import tensorflow as tf

    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(227, 227, 3))

    towerCount = DeepOptions.towerCount
    mult = DeepOptions.netSizeMult
    additLayerCounts = DeepOptions.additLayerCounts
    towerWeightsKerasVar = tf.compat.v1.Variable(np.ones([towerCount * 2 + 2]) * 5 / 9,
                                                 dtype=tf.float32, name='tower_weights')
    towerWeightsKerasVar._trainable = False    # "Doesn't help against Tensor.op is meaningless when eager execution is enabled."
    towerWeights = Input(shape=(towerCount, ), tensor=towerWeightsKerasVar)
        # Looks like input, should be declared in model's input, but (if the tensor parameter is present)
        # should not be supplied to the net (in model.fit and so on)
    # towerWeights = keras.layers.Layer(name='tower_weights')
    # towerWeights.add_weight(shape=[towerCount], trainable=False, initializer='ones')
    # towerWeights = Input(tensor=K.variable(np.ones([4, 1, 1]), name="ones_variable"))

    conv_1 = ModelBlock2(mult * 8, 7, strides=(2, 2), activation='relu',
                    name='conv_11', doubleSizeLayerNames=doubleSizeLayerNames)(inputs)
    conv_1 = fractional_max_pool(1.5)(conv_1)

    conv_1s = []     # Lists only for creating concatenated level for simple data extraction
    add_23s = []
    last_tower_convs = []
    for towerInd in range(towerCount):
        t_conv_2 = conv_1
        t_conv_2 = cross_channel_normalization(name='cross_chan_norm_12%d' % towerInd)(t_conv_2)
        t_conv_2 = BatchNormalization()(t_conv_2)
        t_conv_2 = split_tensor(axis=3, ratio_split=towerCount, id_split=towerInd)(t_conv_2)
        t_conv_2 = ModelBlock2(mult * 16, 5, strides=(2, 2),
                          activation='relu',
                          name='conv_12_%d' % towerInd, doubleSizeLayerNames=doubleSizeLayerNames,
                          kernel_initializer=MyVarianceScalingInitializer(1.0/16 * (towerInd + 1)))(t_conv_2)
            # Source pixels: 11111111222333     Output size - roughly 35 * 35
            # # 111112233
            # #     333334455
        t_conv_2 = SpatialDropout2D(0.5)(t_conv_2)

        if additLayerCounts[0] < 1:
            last_tower_conv = t_conv_2
            # With smallerInputAreas == False actually there are no towers here
        else:
            t_conv_3 = BatchNormalization()(t_conv_2)
            # t_conv_3 = Dropout(0.3)(t_conv_2)
            # t_conv_3 = cross_channel_normalization(name='cross_chan_norm_13%d' % towerInd)(t_conv_3)
            # t_conv_3 = ZeroPadding2D((1, 1))(t_conv_3)
            t_conv_3 = ModelBlock2(mult * 16, 3, padding='same', strides=(1, 1),
                              activation='relu' if towerInd == 0 else 'sigmoid', # activity_regularizer=keras.regularizers.l1(1e-6),
                              name='conv_13_%d' % towerInd, doubleSizeLayerNames=doubleSizeLayerNames,
                              kernel_initializer=MyVarianceScalingInitializer(1.0 / 32 if towerInd == 0 else 0.35))(t_conv_3)
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
                                  activation='relu',
                                  name='conv_14_%d' % towerInd, doubleSizeLayerNames=doubleSizeLayerNames,
                                  kernel_initializer=MyVarianceScalingInitializer(1.0 / 128))(t_conv_4)
                    # Output - 4 * 4
                # conv_4s.append(t_conv_4)
                t_conv_4 = Multiply()([t_conv_4, get_tensor_array_element(towerCount * 2)(towerWeights)])
                    # Switches off level 14
                t_conv_4 = Add(name='add_134_%d' % towerInd)([t_conv_3, t_conv_4])
                # t_conv_4 = Multiply()([ReLU()(t_conv_4), get_tensor_array_element(towerInd)(towerWeights)])
                        # K.constant(value=np.ones([1, 1, 1]), dtype='float32')])   # if towerInd in [0, 3] else \
                            # K.constant(value=0, dtype='float32')])
                     # split_tensor(axis=1, ratio_split=towerCount, id_split=towerInd)(towerWeights)])
                last_tower_conv = t_conv_4
                # add_34s.append(t_conv_4)

        last_tower_conv = Multiply()([last_tower_conv, get_tensor_array_element(towerInd)(towerWeights)])
        # last_tower_convs.append(MaxPooling2D((2, 2), strides=(2, 2))(last_tower_conv))
        # last_tower_convs.append(fractional_max_pool(1.42)(last_tower_conv))
        last_tower_convs.append(last_tower_conv)

    conv_2 = Concatenate(axis=3, name='conv_1_adds')(last_tower_convs)
    # conv_2 = cross_channel_normalization(name="cross_chan_norm_1")(conv_2)
    # conv_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
    conv_2 = fractional_max_pool(1.26)(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    if DeepOptions.modelClass == 'ChanMatrixModel':
        # conv_2 = ModelBlock2(mult * 32, 3, strides=(1, 1),
        #                activation='relu', # activity_regularizer=keras.regularizers.l1(1e-6),
        #                name='conv_21', doubleSizeLayerNames=doubleSizeLayerNames)(conv_2)
        #     # Input - 17 * 17

        print(conv_2.shape)                    # TODO: to try 4 and 5
        conv_22 = Reshape((conv_2.shape[1], conv_2.shape[2], mult, conv_2.shape[3] // mult))(conv_2)
        # max_pool_22 = MaxPooling3D((1, 1, 4), name='max_pool_22')(conv_2)       # E.g. (batch, 25, 25, 1, 64)
        # max_pool_22 = Reshape((conv_2.shape[1], conv_2.shape[2], conv_2.shape[4]))(max_pool_22)
        # print(max_pool_22.shape)
        # max_pool_22 = keras.layers.Cropping2D(1)(max_pool_22)
        conv_22 = ModelBlock2_3D(mult * 4, (3, 3, 1),        # E.g. batch, 25, 25, 12, 48
                        name='conv_22_1', doubleSizeLayerNames=doubleSizeLayerNames)(conv_22)
        print('conv_22_1: ', conv_22.shape)

        assert conv_2.shape[3] // mult != mult        # In that case need another idea how to divide
        conv_222 = Reshape((conv_2.shape[1], conv_2.shape[2], conv_2.shape[3] // mult, mult))(conv_2)
        conv_222 = ModelBlock2_3D(mult * 4, (3, 3, 1),       # E.g. batch, 25, 25, 32, 48
                        name='conv_22_2', doubleSizeLayerNames=doubleSizeLayerNames)(conv_222)
        print('conv_22_2: ', conv_222.shape)
        conv_23 = Concatenate(axis=3, name='concat_23')([conv_22, conv_222])

        # conv_23 = Reshape((conv_23.shape[1], conv_23.shape[2], conv_23.shape[3] * conv_23.shape[4]))(conv_23)
        # l = [max_pool_22, conv_23]
        # if 'conv_23' in doubleSizeLayerNames:
        #     l = [max_pool_22] + l
        # conv_23 = Concatenate(axis=3, name='concat_23')(l)

        conv_24 = Reshape((conv_23.shape[1], conv_23.shape[2],
                           conv_23.shape[3] * conv_23.shape[4]))(conv_23)
        conv_24 = fractional_max_pool(1.3)(conv_24)
        conv_24 = BatchNormalization()(conv_24)              # E.g. batch, 19, 19, 44, 48
        conv_24 = Reshape((conv_24.shape[1], conv_24.shape[2],
                           conv_23.shape[3] // 2, conv_23.shape[4] * 2))(conv_24)
        conv_241 = ModelBlock2_3D(mult * 4, (3, 3, 1),
                        name='conv_24_1', doubleSizeLayerNames=doubleSizeLayerNames)(conv_24)
        print('conv_24_1: ', conv_241.shape)

        conv_242 = Reshape((conv_24.shape[1], conv_24.shape[2], 12,
                            (conv_24.shape[3] * conv_24.shape[4]) // 12))(conv_24)   # E.g. batch, 19, 19, 96, 14
        conv_242 = ModelBlock2_3D(mult * 4, (3, 3, 1),                               # batch, 19, 19, 96, 48
                        name='conv_24_2', doubleSizeLayerNames=doubleSizeLayerNames)(conv_242)
        print('conv_24_2: ', conv_242.shape)

        # conv_25 = Concatenate(axis=3, name='concat_25')([conv_241, conv_242])
        conv_251 = Reshape((conv_241.shape[1], conv_241.shape[2],
                            (conv_241.shape[3] * conv_241.shape[4])))(conv_241)
        conv_251 = ModelBlock2(mult * 32, 2, strides=(1, 1), activation='relu',
                        name='conv_25_1', doubleSizeLayerNames=doubleSizeLayerNames)(conv_251)

        conv_252 = Reshape((conv_242.shape[1], conv_242.shape[2],
                            (conv_242.shape[3] * conv_242.shape[4])))(conv_242)
        conv_252 = ModelBlock2(mult * 32, 2, strides=(1, 1), activation='relu',
                        name='conv_25_2', doubleSizeLayerNames=doubleSizeLayerNames)(conv_252)

    conv_3 = Concatenate(axis=3, name='concat_3')([conv_251, conv_252])
    conv_3 = fractional_max_pool(1.25)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = ModelBlock2(mult * 12, 3, strides=(1, 1), activation='sigmoid',
                        name='conv_3', doubleSizeLayerNames=doubleSizeLayerNames)(conv_3)

    conv_last = conv_3
    # conv_last = BatchNormalization()(conv_last)
    # conv_last = cross_channel_normalization(name='cross_chan_norm_2u')(conv_last)
    # conv_last = MaxPooling2D((2, 2), strides=(2, 2))(conv_last)
    conv_last = SpatialDropout2D(0.4)(conv_last)
    conv_last = fractional_max_pool(1.3)(conv_last)
    conv_last = ModelBlock2(mult * 24, 3, strides=(1, 1), activation='relu',
                           name='conv_4', doubleSizeLayerNames=doubleSizeLayerNames)(conv_last)
    # conv_2 = DepthwiseConv2D(3, depth_multiplier=4, activation='relu', name='conv_2')(conv_1)

    # conv_next = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
    # conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    # conv_2 = ZeroPadding2D((2, 2))(conv_2)
    # conv_next = Conv2D(20, 3, strides=1, activation='relu', name='conv_3')(conv_next)
    # conv_next = DepthwiseConv2D(3, depth_multiplier=2, activation='relu', name='conv_3')(conv_next)

    # dense_1 = MaxPooling2D((2, 2), strides=(2, 2))(conv_last)
    dense_1 = fractional_max_pool(1.3)(conv_last)
    dense_1 = BatchNormalization()(dense_1)
    # dense_1 = Dropout(0.3)(conv_3)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(mult * 8, activation='relu', # activity_regularizer=keras.regularizers.l1(1e-6),
                    name='dense_1')(dense_1)

    # dense_2 = BatchNormalization()(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(mult * 16, name='dense_2')(dense_2)

    # dense_3 = BatchNormalization()(dense_2)
    dense_3 = Dropout(0.3)(dense_2)
    dense_3 = Dense(classCount, name='dense_3')(dense_3)     # Class count
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
        # Maybe layers like conv_22, _23, _24 are above, maybe _24, _25, _26. It's simpler to unite all existing
        name = 'conv_%d2' % (blockInd + 1)
        model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
        name = 'conv_%d4' % (blockInd + 1)
        model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
        if additLayerCounts[blockInd] >= 1:
            name = 'conv_%d3' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d23' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'conv_%d5' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d45' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
        if additLayerCounts[blockInd] >= 2:
            name = 'conv_%d4' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d34' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'conv_%d6' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)
            name = 'add_%d56' % (blockInd + 1)
            model.debug_layers[name] = concatenateLayersByNameBegin(model, name)

    return model

