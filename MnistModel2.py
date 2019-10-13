import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge, Concatenate
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam, SGD
# from keras import backend as K
# from keras.utils.layer_utils import convert_all_kernels_in_model
# from keras.utils import plot_model
import keras.callbacks

# from alexnet_utils import preprocess_image_batch

# from alexnet_additional_layers import split_tensor, cross_channel_normalization
# from decode_predictions import decode_classnames_json, decode_classnumber


# MNIST model like CMnistRecognitionNet.MyModel, but made in the same style as alexnet
def CMnistModel2(weights_path=None):
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')
    inputs = Input(shape=(28, 28, 1))

    conv_1 = Conv2D(80, 5, strides=(2, 2), activation='relu', name='conv_1')(inputs)

    # conv_2 = Conv2D(32, 3, strides=(1, 1), activation='relu', name='conv_2')(conv_1)
    conv_2 = DepthwiseConv2D(3, depth_multiplier=4, activation='relu', name='conv_2')(conv_1)

    conv_next = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)
    # conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    # conv_2 = ZeroPadding2D((2, 2))(conv_2)
    # conv_next = Conv2D(20, 3, strides=1, activation='relu', name='conv_3')(conv_next)
    conv_next = DepthwiseConv2D(3, depth_multiplier=2, activation='relu', name='conv_3')(conv_next)

    dense_1 = Dropout(0.4)(conv_next)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(128, activation='relu', name='dense_1')(dense_1)

    dense_2 = Dropout(0.4)(dense_1)
    dense_2 = Dense(10, name='dense_2')(dense_2)
    prediction = Activation("softmax", name="softmax")(dense_2)

    m = Model(input=inputs, output=prediction)
    print(m.summary())

    if not weights_path is None:
        m.load_weights(weights_path)

    # convert_all_kernels_in_model(m)

    # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(learning_rate=0.001, decay=1e-5)
    # m.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    return m


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
        with self.train_writer.as_default():
            # logs example: {'accuracy': 1.0, 'size': 32, 'loss': 0.00013134594, 'batch': 0}
            tf.summary.scalar('loss_callback', logs.get('loss'), step=self.trainIterNum)
            # print("Callback results %d: %.6f" % (self.trainIterNum, logs.get('loss')))
            # if 'lr' in logs:
            #     tf.summary.scalar('learn_rate', logs.get('lr'), step=self.trainIterNum)
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
