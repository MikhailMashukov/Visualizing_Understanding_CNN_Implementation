# My TensorFlow\TensorWatch\mnist_watch.py  
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
# import subprocess
import sys
import time

import tensorflow as tf
import tensorwatch as tw
import numpy as np
import psutil

# client = tw.WatcherClient(filename='MnistLogs/watch.log')
# stream2 = client.create_stream(expr='lambda d: (d.trainIterNum, np.std(d.d2weights[0]))')
# stream4 = client.create_stream(expr='lambda d: (d.trainIterNum, np.std(d.grad))')
# line_plot2 = tw.Visualizer(stream2, vis_type='line')
# line_plot4 = tw.Visualizer(stream4, vis_type='line', host=line_plot2)
# line_plot2.show()

FLAGS = None

def setProcessPriorityLow():
    p = psutil.Process()
    p.nice(psutil.IDLE_PRIORITY_CLASS)


def weightsTo3D(weights):
    shape = weights.shape
    arr = np.arange(1, shape[0] + 1)
    grid = np.meshgrid(arr, np.arange(1, shape[1] + 1))
        # np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
    coords = np.vstack([grid[1].flatten(), grid[0].flatten(), weights.transpose().flatten()])
    return coords

# print(weightsTo3D(np.random.rand(8, 3)))

class CMnistDataset:
    class TSubset:
        # labels, images
        pass

    def __init__(self):
        self.train = CMnistDataset.TSubset()
        self.test  = CMnistDataset.TSubset()
        (self.train.images, self.train.labels), \
                (self.test.images, self.test.labels) = tf.keras.datasets.mnist.load_data()

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


class MnistRecognitionNet_Simple:
    def __init__(self):
        self.mnist = None

    def init(self):
        self.mnist = CMnistDataset()

        # (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        self.mnist.train.images = self.mnist.train.images / np.float32(255.0)
        self.mnist.test.images  = self.mnist.test.images  / np.float32(255.0)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def run(self):
        self.init()

        self.model.fit(self.mnist.train.images, self.mnist.train.labels, epochs=5)
        self.model.evaluate(self.mnist.test.images, self.mnist.test.labels)


class MnistRecognitionNet:   # Advanced variant
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MnistRecognitionNet.MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation='relu')
            self.d2 = tf.keras.layers.Dense(10, activation='softmax')

            with tf.compat.v1.name_scope('weights'):
                weights = self.d1.get_weights()
                variable_summaries(weights)

        def call(self, x):
            # x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            return self.d2(x)

    def __init__(self):
        self.mnist = None

    def init(self):
        self.mnist = CMnistDataset()

        # (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        self.mnist.train.images = self.mnist.train.images / np.float32(255.0)
        self.mnist.test.images  = self.mnist.test.images  / np.float32(255.0)

        # Add a channels dimension
        self.mnist.train.images = self.mnist.train.images[..., tf.newaxis]
        self.mnist.test.images  = self.mnist.test.images[..., tf.newaxis]

        self.train_ds = tf.data.Dataset.from_tensor_slices(
                (self.mnist.train.images, self.mnist.train.labels)).shuffle(5000).batch(32)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
                (self.mnist.test.images, self.mnist.test.labels)).batch(32)

        # self.train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=1)
        self.train_writer = tf.summary.create_file_writer(FLAGS.log_dir + '/train')
        self.test_writer  = tf.summary.create_file_writer(FLAGS.log_dir + '/test')
        self.watcher = tw.Watcher(filename=FLAGS.log_dir + '/watch.log')
        self.watch_stream = self.watcher.create_stream(name='metric1')
        self.watcher.make_notebook()

        self.model = MnistRecognitionNet.MyModel()

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.gradTfVar = tf.Variable(np.zeros([128, 10], dtype=np.float32), name='grad_var')

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

        # return gradients[0]

    @tf.function
    def test_step(self, images, labels, trainIterNum):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def run(self):
        c_timeMeasureGroupSize = 20

        self.init()

        print('Starting epoch 1')
        trainIterNum = 0
        groupStartTime = datetime.datetime.now()
        for epochNum in range(10):
            for images, labels in self.train_ds:
                gradients = self.train_step(images, labels)
                with self.train_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=trainIterNum)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=trainIterNum)
                    self.watch_stream.write((trainIterNum, float(self.train_accuracy.result())))
                # print(self.model.d2.get_weights())
                self.watcher.observe(trainIterNum=trainIterNum,
                                     weights=self.model.d1.get_weights(),
                                     d2weights=self.model.d2.get_weights())    # list [ np.array(128, 10), np.array(10) ]
                                     # grad=gradients)   # self.gradTfVar.eval())

                # print("Iter. {}: acc. {}".format(trainIterNum, float(acc)))
                # print("Iter. %d :   acc. %.3f" % (trainIterNum, acc))

                trainIterNum += 1
                if trainIterNum % c_timeMeasureGroupSize == 0:
                    print("Iter. %d: acc. %.3f, last %d iter.: %.4f s" %
                          (trainIterNum, self.train_accuracy.result(),
                           c_timeMeasureGroupSize,
                           (datetime.datetime.now() - groupStartTime).total_seconds()))
                    groupStartTime = datetime.datetime.now()

                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels, trainIterNum)

            # print("Iter. %d: acc. %.3f" % (trainIterNum, self.train_accuracy.result()))
            print('Epoch %d, Loss: %.6f, Accuracy: %.3f, Test Loss: %.6f, Test Accuracy: %.3f' %
                  (epochNum + 1,
                   self.train_loss.result(),
                   self.train_accuracy.result() * 100,
                   self.test_loss.result(),
                   self.test_accuracy.result() * 100))

            # self.train_accuracy.reset_states()


if __name__ == '__main__':
    setProcessPriorityLow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=100001,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='MnistLogs', # '/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()


    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    net = MnistRecognitionNet()
    net.run()