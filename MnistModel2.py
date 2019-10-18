import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge, Add, Concatenate, Multiply
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
import keras.layers

from keras.optimizers import Adam, SGD
from keras import backend as K
# from keras.utils.layer_utils import convert_all_kernels_in_model
# from keras.utils import plot_model
import keras.callbacks
import keras.initializers

# from alexnet_utils import preprocess_image_batch

from keras.layers.core import Lambda
from alexnet_additional_layers import split_tensor, cross_channel_normalization
# from decode_predictions import decode_classnames_json, decode_classnumber


# MNIST model like CMnistRecognitionNet.MyModel, but made in the same style as alexnet

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


# MNIST model like CMnistRecognitionNet.MyModel, but made in the same style as alexnet
def CMnistModel2(weights_path=None):
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    conv_1 = Conv2D(32, 5, strides=(2, 2), activation='relu', name='conv_1')(inputs)

    conv_2 = Conv2D(20, 3, strides=(1, 1), activation='relu', name='conv_2')(conv_1)

    conv_next = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
    # conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    # conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_next = Conv2D(20, 3, strides=1, activation='relu', name='conv_3')(conv_next)

    dense_1 = Flatten(name="flatten")(conv_next)
    dense_1 = Dense(128, activation='relu', name='dense_1')(dense_1)

    dense_2 = Dropout(0.4)(dense_1)
    dense_2 = Dense(10, name='dense_2')(dense_2)
    prediction = Activation("softmax", name="softmax")(dense_2)

    model = Model(input=inputs, output=prediction)
    print(model.summary())

    if not weights_path is None:
        model.load_weights(weights_path)

    # convert_all_kernels_in_model(model)

    # # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(learning_rate=0.001, decay=1e-5)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model

def get_tensor_array_element(x):
    def f(X):                               # Here we have e.g. np.ones([4])
        return X[x : x + 1]

    def g(input_shape):
        output_shape = list(input_shape)    # Here - e.g. (None, 4)
        output_shape[-1] = 1
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape))

def concatenateLayersByNameBegin(model, nameBegin):
    foundLayers = []
    l = len(nameBegin)
    for layer in model.layers:
        if layer.name[:l] == nameBegin:
            foundLayers.append(layer.output)
    return Concatenate(axis=3, name=nameBegin)(foundLayers)

def CMnistModel3_Towers(weights_path=None):
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    conv_1 = Conv2D(16, 5, strides=(2, 2), activation='relu', name='conv_1_common')(inputs)

    towerCount = 4
    additLayerCount = 2
    towerWeightsKerasVar = K.variable(np.ones([towerCount]), name='tower_weights')
    towerWeights = Input(shape=(towerCount, ), tensor=towerWeightsKerasVar)
        # Looks like input, should be declared in model's input, but (if the tensor parameter is present)
        # should not be supplied to the net (in model.fit and so on)
    # towerWeights = keras.layers.Layer(name='tower_weights')
    # towerWeights.add_weight(shape=[towerCount], trainable=False, initializer='ones')
    # towerWeights = Input(tensor=K.variable(np.ones([4, 1, 1]), name="ones_variable"))

    conv_1s = []     # Lists only for creating concatenated level for simple data extraction
    # conv_2s = []
    add_23s = []
    # conv_3s = []
    # add_34s = []
    # conv_4s = []
    last_tower_convs = []
    for towerInd in range(towerCount):
        x0 = (towerInd % 2) * 12
        y0 = ((towerInd // 2) % 2) * 12
        # t_conv_1 = Conv2D(8, 5, strides=(2, 2), activation='relu', name='conv_1_%d' % towerInd)(
        #         inputs[:, x0 : x0 + 16, y0 : y0 + 16, :])
        t_input = cut_image_tensor(x0, x0 + 16, y0, y0 + 16)(inputs)
        t_conv_1 = Conv2D(8, 5, strides=(2, 2), activation='relu', name='conv_1_%d' % towerInd)(t_input)
        conv_1s.append(t_conv_1)

        cut_main_conv_1 = cut_image_tensor(x0 // 2, x0 // 2 + 6, y0 // 2, y0 // 2 + 6)(conv_1)
            # Output - 6 * 6

        # common_conv_x0 = (towerInd % 2) * 6
        # union = Concatenate(axis=3)([conv_1[:, x0 // 2 : x0 // 2 + 6, y0 // 2 : y0 // 2 + 6, :]])  # t_conv_1])
        union = Concatenate(axis=3)([cut_main_conv_1, t_conv_1])
        union_norm = BatchNormalization()(union)
        t_conv_2 = Conv2D(8, 3, strides=(1, 1), activation='relu', name='conv_2_%d' % towerInd)(union_norm)
            # Output - 4 * 4
        # t_conv_3 = MaxPooling2D((2, 2), strides=(2, 2))(t_conv_2)
        # conv_2s.append(t_conv_2)

        if additLayerCount < 1:
            last_tower_convs.append(t_conv_2)
        else:
            # t_conv_3 = BatchNormalization()(t_conv_2)
            t_conv_3 = Dropout(0.3)(t_conv_2)
            # t_conv_3 = ZeroPadding2D((1, 1))(t_conv_3)
            t_conv_3 = Conv2D(8, 3, padding='same', strides=(1, 1),
                              activation='relu', name='conv_3_%d' % towerInd,
                              kernel_initializer=MyInitializer)(t_conv_3)
                # Output - 4 * 4
            # conv_3s.append(t_conv_3)
            t_conv_3 = Add(name='add_23_%d' % towerInd)([t_conv_2, t_conv_3])
            add_23s.append(t_conv_3)
        # # t_conv_3 = MaxPooling2D((2, 2), strides=(2, 2))(t_conv_2)

        if additLayerCount < 2:
            last_tower_convs.append(t_conv_3)
        else:
            t_conv_4 = BatchNormalization()(t_conv_3)
            # t_conv_4 = ZeroPadding2D((1, 1))(t_conv_4)
            t_conv_4 = Conv2D(8, 3, padding='same', strides=(1, 1),
                              # activation='relu',
                              name='conv_4_%d' % towerInd,
                              kernel_initializer=MyInitializer)(t_conv_4)
                # Output - 4 * 4
            # conv_4s.append(t_conv_4)
            t_conv_4 = Add(name='add_34_%d' % towerInd)([t_conv_3, t_conv_4])
            t_conv_4 = Multiply()([ReLU()(t_conv_4), get_tensor_array_element(towerInd)(towerWeights)])
                    # K.constant(value=np.ones([1, 1, 1]), dtype='float32')])   # if towerInd in [0, 3] else \
                        # K.constant(value=0, dtype='float32')])
                 # split_tensor(axis=1, ratio_split=towerCount, id_split=towerInd)(towerWeights)])
            last_tower_convs.append(t_conv_4)
            # add_34s.append(t_conv_4)

    conv_last = Concatenate(axis=3, name='conv_%d_adds' % (2 + additLayerCount))(last_tower_convs)
        # This layer produces 8*1 gradients tensor, so special conv_2/3 was added
    conv_last = Conv2D(32, 1, strides=(1, 1), activation='relu',
                       name='conv_%d' % (3 + additLayerCount))(conv_last)

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
    dense_1 = Dense(64, activation='relu', name='dense_1')(dense_1)

    # dense_2 = BatchNormalization()(dense_1)
    dense_2 = Dropout(0.3)(dense_1)
    dense_2 = Dense(10, name='dense_2')(dense_2)
    prediction = Activation("softmax", name="softmax")(dense_2)

    model = Model(inputs=[inputs, towerWeights], outputs=prediction)
    print(model.summary())

    if not weights_path is None:
        model.load_weights(weights_path)

    # convert_all_kernels_in_model(model)

    # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(learning_rate=0.001, decay=1e-5)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    model.variables = {'towers_weights': towerWeightsKerasVar}

    model.debug_layers = dict()
    model.debug_layers['conv_1'] = Concatenate(axis=3, name='conv_1')(conv_1s)
    model.debug_layers['conv_2'] = concatenateLayersByNameBegin(model, 'conv_2')
    if additLayerCount >= 1:
        model.debug_layers['conv_3'] = concatenateLayersByNameBegin(model, 'conv_3')
    if additLayerCount >= 2:
        model.debug_layers['conv_4'] = concatenateLayersByNameBegin(model, 'conv_4')
        model.debug_layers['add_23'] = concatenateLayersByNameBegin(model, 'add_23')

    return model


def MyInitializer(shape, dtype=None):
    v = keras.initializers.VarianceScaling(scale=1. / 16,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=None)
    return v(shape, dtype)

def CMnistModel4_Matrix(weights_path=None):
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    conv_1 = Conv2D(32, 5, strides=(2, 2), activation='relu', name='conv_1_all')(inputs)

    matrixWidth = 5
    matrixHeight = 3
    additLayerPairCount = 1
    inputBlockWeightsKerasVar = K.variable(np.ones([matrixWidth + matrixHeight]), name='tower_weights')
    inputBlockWeights = Input(shape=(matrixWidth + matrixHeight, ), tensor=inputBlockWeightsKerasVar)

    conv_1_outputs = [split_tensor(axis=3, ratio_split=matrixWidth + matrixHeight, id_split=i)(conv_1) \
                      for i in range(matrixWidth + matrixHeight)]           # Each - 12 * 12 * 4 channels
    for i in range(matrixWidth + matrixHeight):
        conv_1_outputs[i] = BatchNormalization(name='conv_1_%d' % i)(conv_1_outputs[i])
        conv_1_outputs[i] = Multiply()([conv_1_outputs[i], get_tensor_array_element(i)(inputBlockWeights)])

    conv_2_matrix = []
    for x in range(matrixWidth):
        conv_2_matrix.append([None] * matrixHeight)
        for y in range(matrixHeight):
            conv_2_input = Concatenate(axis=3)([conv_1_outputs[x], conv_1_outputs[matrixWidth + y]])
            conv = Conv2D(4, 3, strides=(1, 1), activation='relu', name='conv_2_%d_%d' % (x, y))(conv_2_input)
                # Each - 10 * 10
            # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
                # Each - 5 * 5, totally were 6 * 6 * 8 channels (with matrixWidth and 48 conv_1 channels)
            # conv = BatchNormalization()(conv)
            # conv = Dropout(0.3)(conv)

            conv_2_matrix[x][y] = conv

    conv_3s_horiz = []
    conv_3s_vert = []
    conv_3s_horiz_before_zp = []
    conv_3s_vert_before_zp = []
    for i in range(matrixHeight):
        conv_3_input = Concatenate(axis=3)([conv_2_matrix[j][i] for j in range(matrixWidth)])
            # Each - 5 * 5 * 48 channels
        conv = Conv2D(4, 3, strides=(1, 1), activation='relu', name='conv_3_horiz_%d' % (i))(conv_3_input)
            # 3 * 3
        conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
            # 2 * 2, 8 channels each, 96 channels total
        conv = BatchNormalization()(conv)
        # conv = keras.layers.Add()([conv_1[], conv])
        conv_3s_horiz_before_zp.append(conv)
        conv = ZeroPadding2D((1, 1))(conv)
        conv_3s_horiz.append(conv)      # Convolutions of horizontal rows of the matrix

    for i in range(matrixWidth):
        conv_3_input = Concatenate(axis=3)([conv_2_matrix[i][j] for j in range(matrixHeight)])
        conv = Conv2D(4, 3, strides=(1, 1), activation='relu', name='conv_3_vert_%d' % (i))(conv_3_input)
        conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
        conv = BatchNormalization()(conv)
        conv_3s_vert_before_zp.append(conv)
        conv = ZeroPadding2D((1, 1))(conv)
        conv_3s_vert.append(conv)
        conv_3_all = Concatenate(axis=3, name='conv_3')(conv_3s_vert_before_zp + conv_3s_horiz_before_zp)

    if not additLayerPairCount:
        dense_1 = Concatenate(axis=3)(conv_3s_horiz_before_zp + conv_3s_vert_before_zp)
    else:      # Two more layers
        conv_next_matrix = []
        for x in range(matrixWidth):
            conv_next_matrix.append([None] * matrixHeight)
            for y in range(matrixHeight):
                input = Concatenate(axis=3)([conv_3s_horiz[y], conv_3s_vert[x]])
                conv = Conv2D(4, 3, strides=(1, 1), activation='relu', name='conv_4_%d_%d' % (x, y))(input)
                # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
                conv = BatchNormalization()(conv)
                # conv = Dropout(0.3)(conv)

                conv_next_matrix[x][y] = conv

        conv_next_horiz = []
        conv_next_vert = []
        for i in range(matrixHeight):
            input = Concatenate(axis=3)([conv_next_matrix[j][i] for j in range(matrixWidth)])
                # # Each - 5 * 5 * 48 channels
            conv = Conv2D(4, 3, padding='same', strides=(1, 1), activation='relu', name='conv_5_horiz_%d' % (i),
                    kernel_initializer=MyInitializer)(input)
            conv = keras.layers.Add()([conv_3s_horiz_before_zp[i], conv])
            # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
            conv = Multiply()([conv, get_tensor_array_element(matrixWidth + i)(inputBlockWeights)])
            conv_next_horiz.append(conv)
            # prediction = Dense(10, name='dense_2')(Flatten(name="flatten")  (conv))
            # return Model(input=inputs, output=prediction)

        for i in range(matrixWidth):
            input = Concatenate(axis=3)([conv_next_matrix[i][j] for j in range(matrixHeight)])
            conv = Conv2D(4, 3, padding='same', strides=(1, 1), activation='relu', name='conv_5_vert_%d' % (i),
                    kernel_initializer=MyInitializer)(input)
            conv = keras.layers.Add()([conv_3s_vert_before_zp[i], conv])
            # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
                # # 2 * 2, 8 channels each, 96 channels total
            conv = Multiply()([conv, get_tensor_array_element(i)(inputBlockWeights)])
            conv_next_vert.append(conv)
        dense_1 = Concatenate(axis=3, name='conv_5')(conv_next_vert + conv_next_horiz)

    dense_1 = BatchNormalization()(dense_1)
    # dense_1 = Dropout(0.3)(dense_1)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(128, activation='relu', name='dense_1')(dense_1)

    dense_2 = BatchNormalization()(dense_1)
    # dense_2 = Dropout(0.3)(dense_1)
    dense_2 = Dense(10, name='dense_2')(dense_2)
    prediction = Activation("softmax", name="softmax")(dense_2)

    model = Model(input=[inputs, inputBlockWeights], output=prediction)
    print(model.summary())

    if not weights_path is None:
        model.load_weights(weights_path)

    # convert_all_kernels_in_model(model)

    # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(learning_rate=0.001, decay=1e-5)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    model.variables = {'towers_weights': inputBlockWeightsKerasVar}

    model.debug_layers = dict()
    model.debug_layers['conv_1'] = concatenateLayersByNameBegin(model, 'conv_1')
    model.debug_layers['conv_2'] = concatenateLayersByNameBegin(model, 'conv_2')
    model.debug_layers['conv_3'] = concatenateLayersByNameBegin(model, 'conv_3')
    if additLayerPairCount > 0:
        model.debug_layers['conv_4'] = concatenateLayersByNameBegin(model, 'conv_4')
        model.debug_layers['conv_5'] = concatenateLayersByNameBegin(model, 'conv_5')

    return model


# This class is here instead of MnistNet.py in order to allow lazy importing.
# It writes summary and also passes signals about batch end to outer classes
class CSummaryWriteCallback(keras.callbacks.Callback):
    # Callback - like CBaseLearningCallback
    def __init__(self, mnistDataset, train_writer, test_writer, initialIterNum, learningCallback):
        self.mnistDataset = mnistDataset
        self.train_writer = train_writer
        self.test_writer = test_writer
        self.initialIterNum = initialIterNum
        self.trainIterNum = initialIterNum
        self.learningCallback = learningCallback

        fullTestDataset = self.mnistDataset.getNetSource('test')
        testDatasetSize = 1000
        self.testDataset = (fullTestDataset[0][:testDatasetSize],
                   tf.keras.utils.to_categorical(fullTestDataset[1][:testDatasetSize]))

    # def on_train_begin(self, logs={}):
    #     self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.trainIterNum += 1

        # self.losses.append(logs.get('loss'))
        # with self.train_writer.as_default():
        #     # logs example: {'accuracy': 1.0, 'size': 32, 'loss': 0.00013134594, 'batch': 0}
        #     tf.summary.scalar('loss_callback', logs.get('loss'), step=self.trainIterNum)
        #     # print("Callback results %d: %.6f" % (self.trainIterNum, logs.get('loss')))
        #     # if 'lr' in logs:
        #     #     tf.summary.scalar('learn_rate', logs.get('lr'), step=self.trainIterNum)
        if self.trainIterNum % 50 == 0:
            passed = self.trainIterNum < self.initialIterNum
            if passed <= 250 or self.trainIterNum % 200 == 0:
                with self.train_writer.as_default():
                    tf.summary.scalar('learn_rate', keras.backend.eval(self.model.optimizer.lr), \
                                      step=self.trainIterNum)
                with self.test_writer.as_default():
                    scores = self.model.evaluate(self.testDataset[0], self.testDataset[1], verbose=1)
                    tf.summary.scalar('loss_callback', scores[0], step=self.trainIterNum)
                    tf.summary.scalar('accuracy_callback', scores[1], step=self.trainIterNum)
        self.learningCallback.onBatchEnd(self.trainIterNum, logs)
