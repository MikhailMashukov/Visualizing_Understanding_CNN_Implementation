# Part of MnistVis.py, extracted to make lazy import tensorflow properly
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import datetime
import math
import numpy as np
# import psutil
# import subprocess
# import sys
# import time

import MnistModel2

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
      mean = tf.reduce_mean(input_tensor=var)
      tf.compat.v1.summary.scalar('mean', mean)
      with tf.compat.v1.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
      tf.compat.v1.summary.scalar('stddev', stddev)
      tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
      tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
      tf.compat.v1.summary.histogram('histogram', var)


# Net as in Google tutorial for experts
class CMnistRecognitionNet:
    class MyModel(tf.keras.Model):
        # Error in loadState: You are trying to load a weight file containing 4 layers into a model with 0 layers.

        def __init__(self, highestLayerName=None):
            super(CMnistRecognitionNet.MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu', name='conv_1')
            self.maxPool = tf.keras.layers.MaxPooling2D((2, 2))
            self.conv2 = tf.keras.layers.Conv2D(20, 3, activation='relu', name='conv_2')
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
            self.d2 = tf.keras.layers.Dense(10, activation='softmax', name='dense_2')

            # with tf.compat.v1.name_scope('weights'):
            #     weights = self.d1.get_weights()
            #     variable_summaries(weights)

        def call(self, x, highestLayerName=None):
            x = self.conv1(x)
            if highestLayerName == 'conv_1':
                return x
            x = self.maxPool(x)
            x = self.conv2(x)
            x = self.flatten(x)
            x = self.d1(x)
            if highestLayerName == 'dense_1':
                return x
            x = tf.keras.layers.Dropout(0.2)(x)
            return self.d2(x)


    def __init__(self):
        self.mnist = None
        self.timeMeasureGroupSize = 20
        self.createModel()

    def init(self, mnistDataset, logDir):
        self.mnist = mnistDataset
        self.logDir = logDir

        # (x_train, y_train), (x_test, y_test) = self.mnist.load_data()

        # Add a channels dimension
        # self.mnist.train.images = self.mnist.train.images[..., tf.newaxis]
        # self.mnist.test.images  = self.mnist.test.images[..., tf.newaxis]

        self.train_ds = tf.data.Dataset.from_tensor_slices(
                self.mnist.getNetSource('train')).shuffle(5000).batch(32)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
                self.mnist.getNetSource('test')).batch(32)

        # self.train_writer = tf.compat.v1.summary.FileWriter(self.logDir + '/train', sess.graph)
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logDir, histogram_freq=1)
        self.train_writer = tf.summary.create_file_writer(self.logDir + '/train')
        self.test_writer  = tf.summary.create_file_writer(self.logDir + '/test')
        # self.watcher = tw.Watcher(filename=self.logDir + '/watch.log')
        # self.watch_stream = self.watcher.create_stream(name='metric1')
        # self.watcher.make_notebook()

        # self.createModel()

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.gradTfVar = tf.Variable(np.zeros([128, 10], dtype=np.float32), name='grad_var')

        self.trainIterNum = 0

    def createModel(self):
        self.model = CMnistRecognitionNet.MyModel()
        inputs = tf.keras.Input(shape=(28, 28, ), name='img')
        self.kerasModel = tf.keras.Model(inputs=inputs, outputs=self.model.d2, name='mnist_model')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # tf.compat.v1.assign(self.gradTfVar, gradients[2])
        # print(predictions)   # 32 * 10
        # print(loss)          # 32 * 10

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        # acc2 = self.train_accuracy.result()

        # self.tensorboard_callback()

        return gradients

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def doLearning(self, iterCount):
        print("Running %d iteration(s)" % iterCount)
        groupStartTime = datetime.datetime.now()
        restIterCount = iterCount
        while restIterCount > 0:
            for images, labels in self.train_ds:
                gradients = self.train_step(images, labels)
                with self.train_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=self.trainIterNum)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=self.trainIterNum)
                #     self.watch_stream.write((trainIterNum, float(self.train_accuracy.result())))
                # # print(self.model.d2.get_weights())
                # self.watcher.observe(trainIterNum=trainIterNum,
                #                      weights=self.model.d1.get_weights(),
                #                      d2weights=self.model.d2.get_weights())    # list [ np.array(128, 10), np.array(10) ]
                #                      # grad=gradients)   # self.gradTfVar.eval())

                # print("Iter. {}: acc. {}".format(trainIterNum, float(acc)))
                # print("Iter. %d :   acc. %.3f" % (trainIterNum, acc))

                self.trainIterNum += 1
                restIterCount -= 1
                if restIterCount <= 0:
                    break
                # if self.trainIterNum % self.timeMeasureGroupSize == 0:

        infoStr = "Iter. %d: loss %.5f, acc. %.4f, last %d iter.: %.4f s" % \
                  (self.trainIterNum,
                   self.train_loss.result(), self.train_accuracy.result(),
                   iterCount,
                   (datetime.datetime.now() - groupStartTime).total_seconds())
                # groupStartTime = datetime.datetime.now()

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        return infoStr

    def runFullTrain(self):
        # self.init()

        print('Starting epoch 1')
        # trainIterNum = 0
        for epochNum in range(10):
            self.runOneEpoch()
            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            # print("Iter. %d: acc. %.3f" % (trainIterNum, self.train_accuracy.result()))
            print('Epoch %d, Loss: %.6f, Accuracy: %.3f, Test Loss: %.6f, Test Accuracy: %.3f' %
                  (epochNum + 1,
                   self.train_loss.result(),
                   self.train_accuracy.result() * 100,
                   self.test_loss.result(),
                   self.test_accuracy.result() * 100))

            # self.train_accuracy.reset_states()


# Net based on model, made in the same style as alexnet
class CMnistRecognitionNet2(CMnistRecognitionNet):
    def __init__(self, highest_layer=None, base_model=None):
        self.highest_layer = highest_layer
        # if base_model is None:
        #     print("None")
        self.base_model = base_model
        self.batchSize = 64
        super(CMnistRecognitionNet2, self).__init__()

    def createModel(self):
        if not self.base_model:
            # self.base_model = MnistModel2.CMnistModel2()   # If no base_model, create net
            self.base_model = MnistModel2.CMnistModel4_Matrix()
        self.model = self._sub_model() if self.highest_layer else self.base_model         # Use full network if no highest_layer

    def _sub_model(self):
        from keras.models import Model

        if isinstance(self.highest_layer, int):
            highest_layer_name = 'conv_{}'.format(self.highest_layer)
        else:
            highest_layer_name = self.highest_layer
            # if highest_layer_name[-len('_weights') : ] == '_weights':
            #     highest_layer = self.base_model.get_layer(highest_layer_name[ : -len('_weights')])._trainable_weights
            # else:
        try:
            highest_layer = self.base_model.get_layer(highest_layer_name)
            return Model(inputs=self.base_model.input,
                         outputs=highest_layer.output)
        except ValueError as valueEx:
            highest_layer_output = self.base_model.debug_layers[highest_layer_name]
            return Model(inputs=self.base_model.input,
                         outputs=highest_layer_output)

    # def predict(self, img_path):
    #     img = preprocess_image_batch(img_path)
    #     return self.model.predict(img)
    #

    def doLearning(self, epochCount, learningCallback,
                   epochImageCount=None, initialEpochNum=0):
        from keras.callbacks import TensorBoard

        # epochCount = int(math.ceil(iterCount / 100))
        fullDataset = self.mnist.getNetSource('train')
        fullTestDataset = self.mnist.getNetSource('test')
        fullDatasetImageCount = fullDataset[0].shape[0]
        if epochImageCount is None:
            epochImageCount = fullDatasetImageCount
        print("Running %d epoch(s) from %d, %d images each" % \
                (epochCount, initialEpochNum, epochImageCount))
        groupStartTime = datetime.datetime.now()

        if epochImageCount == fullDatasetImageCount:
            dataset = (fullDataset[0],
                       tf.keras.utils.to_categorical(fullDataset[1]))
        else:
            permut = np.random.permutation(fullDatasetImageCount)[:epochImageCount]
            dataset = (fullDataset[0][permut, :, :, :],
                       tf.keras.utils.to_categorical(fullDataset[1][permut]))

        testDatasetSize = epochImageCount // 6
        if testDatasetSize >= fullTestDataset[0].shape[0]:
            testDataset = (fullTestDataset[0],
                   tf.keras.utils.to_categorical(fullTestDataset[1]))
        else:
            permut = np.random.permutation(fullTestDataset[0].shape[0])[:testDatasetSize]
            testDataset = (fullTestDataset[0][permut, :, :, :],
                           tf.keras.utils.to_categorical(fullTestDataset[1][permut]))
        # testDataset = (fullTestDataset[0][:testDatasetSize],
        #            tf.keras.utils.to_categorical(fullTestDataset[1][:testDatasetSize]))

        tensorBoardCallback = TensorBoard(log_dir='QtLogs', histogram_freq=0,
                write_graph=False, write_grads=False, write_images=1,    # batch_size=32,
                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=True, embeddings_data=None,
                update_freq='epoch')
        summaryCallback = MnistModel2.CSummaryWriteCallback(self.mnist,
                self.train_writer, self.test_writer,
                int(initialEpochNum * fullDataset[0].shape[0] / self.batchSize),
                learningCallback)
        # tensorBoardCallback.set_model(self.model)

        history = self.model.fit(x=dataset[0], y=dataset[1], validation_data=testDataset,
                                 epochs=initialEpochNum + epochCount, initial_epoch=initialEpochNum,
                                 batch_size=self.batchSize, verbose=2, callbacks=[tensorBoardCallback, summaryCallback])

        try:
            if not history.history:
                raise Exception('empty history object')
            infoStr = "loss %.5f" % history.history['loss'][-1]
            infoStr += ", acc %.4f" % history.history['accuracy'][-1]
        except Exception as ex:
            print("Error in doLearning: %s" % str(ex))
            infoStr = ''

        infoStr = "%s, last %d epochs: %.4f s" % \
                  (infoStr, epochCount,
                   (datetime.datetime.now() - groupStartTime).total_seconds())
                # groupStartTime = datetime.datetime.now()

        return infoStr
