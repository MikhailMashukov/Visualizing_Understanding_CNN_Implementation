# Part of MnistVis.py, extracted to make lazy import tensorflow properly
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import datetime
import numpy as np
import psutil
# import subprocess
import sys
import time

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


class CMnistRecognitionNet:
    class MyModel(tf.keras.Model):
        def __init__(self, highestLayerName=None):
            super(CMnistRecognitionNet.MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv_1')
            self.maxPool = tf.keras.layers.MaxPooling2D((2, 2))
            self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv_2')
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
            self.d2 = tf.keras.layers.Dense(10, activation='softmax', name='dense_2')

            with tf.compat.v1.name_scope('weights'):
                weights = self.d1.get_weights()
                variable_summaries(weights)

        def call(self, x, highestLayerName=None):
            x = self.conv1(x)
            if highestLayerName == 'conv_1':
                return x
            x = self.flatten(x)
            x = self.d1(x)
            if highestLayerName == 'dense_1':
                return x
            x = tf.keras.layers.Dropout(0.2)(x)
            return self.d2(x)

    def __init__(self):
        self.mnist = None
        self.timeMeasureGroupSize = 20

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

        self.model = CMnistRecognitionNet.MyModel()

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.gradTfVar = tf.Variable(np.zeros([128, 10], dtype=np.float32), name='grad_var')

        self.trainIterNum = 0

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


    def runOneEpoch(self):
        groupStartTime = datetime.datetime.now()
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
                if self.trainIterNum % self.timeMeasureGroupSize == 0:
                    print("Iter. %d: loss %.5f, acc. %.4f, last %d iter.: %.4f s" %
                          (self.trainIterNum,
                           self.train_loss.result(), self.train_accuracy.result(),
                           self.timeMeasureGroupSize,
                           (datetime.datetime.now() - groupStartTime).total_seconds()))
                    groupStartTime = datetime.datetime.now()

                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

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


