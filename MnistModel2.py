from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import plot_model

# from alexnet_utils import preprocess_image_batch

# from alexnet_additional_layers import split_tensor, cross_channel_normalization
# from decode_predictions import decode_classnames_json, decode_classnumber


# MNIST model like CMnistRecognitionNet.MyModel, but made in the same style as alexnet
def CMnistModel2(weights_path=None):
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    conv_1 = Conv2D(32, 3, strides=(2, 2), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((2, 2), strides=(1, 1))(conv_1)
    # conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    # conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Conv2D(20, 3, strides=1, activation='relu', name='conv_2')(inputs)

    dense_1 = MaxPooling2D((2, 2), strides=(1, 1), name="convpool_2")(conv_2)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(128, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.4)(dense_1)
    dense_2 = Dense(10, name='dense_2')(dense_2)
    prediction = Activation("softmax", name="softmax")(dense_2)

    m = Model(input=inputs, output=prediction)

    if not weights_path is None:
        m.load_weights(weights_path)

    # convert_all_kernels_in_model(m)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    return m
