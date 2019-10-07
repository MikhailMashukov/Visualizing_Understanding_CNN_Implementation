# My TensorFlow\TensorWatch\mnist_watch.py, cut and attached to VisQtMain
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
# import subprocess
import sys
import time

import numpy as np
import psutil

# FLAGS = None


class CMnistVisWrapper:
    def __init__(self):
        self.mnistDataset = CMnistDataset()
        self.net = None

    def getImageDataset(self):
        return self.mnistDataset

    def getNetLayersToVisualize(self):
        return ['conv_1', 'dense_1', 'dense_2']

    def getImageActivations(self, layerName, imageNum):
        # if isinstance(layer, int):
        #     layerName = 'conv_%d' % layer
        # else:
        #     layerName = layer
        # itemCacheName = 'act_%s_%d' % (layerName, imageNum)
        # cacheItem = self.activationCache.getObject(itemCacheName)
        # if not cacheItem is None:
        #     return cacheItem

        if self.net is None:
            import MnistNet

            self.net = MnistNet.CMnistRecognitionNet()  # self.mnistDataset)     net uses its own copy, a bit changed data
            dataset = CMnistDataset()
            # dataset.loadData()
            self.net.init(dataset, 'QtLogs')
        imageData = self.mnistDataset.getImage(imageNum)
        activations = self.net.model.predict(np.expand_dims(np.expand_dims(imageData, 0), 3))
        return activations

    def loadCacheState(self):
        pass

    def saveCacheState(self):
        pass


class CMnistDataset:
    class TSubset:
        # labels, images
        pass

    def __init__(self):
        self.train = None
        self.test = None

    def getImage(self, imageNum, _=None):      # ImageNum here - 1-based
        if self.test is None:
            self.loadData()
        return self.test.images[imageNum - 1]

    def getNetSource(self, type='train'):
        if self.test is None:
            self.loadData()
        subset = self.train if type == 'train' else self.test
        return (np.expand_dims(subset.images, axis=3), subset.labels)

    # If data are not necessary right now, it' ok not to call this.
    # This method is necessary if you want to access self.train/.test directly
    def loadData(self):
        import tensorflow as tf

        self.train = CMnistDataset.TSubset()
        self.test  = CMnistDataset.TSubset()
        (self.train.images, self.train.labels), \
                (self.test.images, self.test.labels) = tf.keras.datasets.mnist.load_data()
        self.train.images = self.train.images / np.float32(255.0)
        self.test.images  = self.test.images  / np.float32(255.0)



