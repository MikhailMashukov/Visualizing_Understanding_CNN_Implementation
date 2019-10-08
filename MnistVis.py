# My TensorFlow\TensorWatch\mnist_watch.py, cut and attached to VisQtMain
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
# import subprocess
import sys
import time

import numpy as np
import psutil

import DataCache
from MyUtils import *

# FLAGS = None

class CMnistVisWrapper:
    def __init__(self):
        self.name = 'mnist'
        self.mnistDataset = CMnistDataset()
        self.net = None
        self.weightsPath = 'QtLogs/MnistWeights.h5'
        self.activationCache = DataCache.CDataCache(64 * getCpuCoreCount())

    def getImageDataset(self):
        return self.mnistDataset

    def getNetLayersToVisualize(self):
        return ['conv_1', 'dense_1', 'dense_2']

    def getImageActivations(self, layerName, imageNum):
        itemCacheName = 'act_%s_%d' % (layerName, imageNum)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        if self.net is None:
            self.initNet()
        imageData = self.mnistDataset.getImage(imageNum)
        imageData = np.expand_dims(imageData, 0)
        activations = self.net.model.predict(imageData)   # np.expand_dims(imageData, 0), 3))
        self.activationCache.saveObject(itemCacheName, activations)
        return activations

    def doLearning(self, iterCount):
        if self.net is None:
            self.initNet()
        infoStr = self.net.doLearning(iterCount)
        self.activationCache.clear()
        return infoStr

    def saveState(self):
        try:
            with open('Data/MnistVisActCache.dat', 'wb') as file:
                self.activationCache.saveState_OpenedFile(file)
            if not self.net is None:
                self.net.model.save_weights(self.weightsPath)
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

    # When it is desirable to initialize quickly (without tensorflow)
    def loadCacheState(self):
        try:
            with open('Data/MnistVisActCache.dat', 'rb') as file:
                self.activationCache.loadState_OpenedFile(file)
        except Exception as ex:
            print("Error in loadCacheState: %s" % str(ex))

    def loadState(self):
        try:
            self.loadCacheState()
            self.initNet()

            # l = self.net.model.get_layer('conv_1')
            # w = l.get_weights()[0]

            self.net.model.load_weights(self.weightsPath)

            # l2 = self.net.model.get_layer('conv_1')
            # w2 = l.get_weights()[0]
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

    def getCacheStatusInfo(self):
        return '%.2f MBs' % \
                (self.activationCache.getUsedMemory() / (1 << 20))


    # If data are not necessary right now, it' ok not to call this.
    def initNet(self):
        import MnistNet

        self.net = MnistNet.CMnistRecognitionNet2()
        dataset = CMnistDataset()
        # dataset.loadData()
        self.net.init(dataset, 'QtLogs')


    @staticmethod
    def get_source_block_calc_func(layerName):
        if layerName == 'conv_1':
            return CMnistVisWrapper.get_conv_1_source_block
        else:
            return CMnistVisWrapper.get_entire_image_block

    # Returns source pixels block, corresponding to the layer conv_1 pixel (x, y)
    @staticmethod
    def get_conv_1_source_block(x, y):
        source_xy_0 = (x, y)
        size = 3
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_entire_image_block(_, y):
        return (0, 0, 28, 28)


class CMnistDataset:
    class TSubset:
        # labels, images
        pass

    def __init__(self):
        self.train = None
        self.test = None
        self.preparedDatasetFileName = 'Data/MnistVisDataset.dat'

    def getImage(self, imageNum, preprocessStage='net'):      # ImageNum here - 1-based
        if self.test is None:
            self.loadData()
        data = np.expand_dims(self.test.images[imageNum - 1], axis=2)
        if preprocessStage == 'cropped':
            data = (data * 255).astype(np.uint8)
        return data

    def getNetSource(self, type='train'):
        if self.test is None:
            self.loadData()
        subset = self.train if type == 'train' else self.test
        return (np.expand_dims(subset.images, axis=3), subset.labels)

    # If data are not necessary right now, it' ok not to call this.
    # This method is necessary if you want to access self.train/.test directly
    def loadData(self):
        import pickle

        try:
            with open(self.preparedDatasetFileName, 'rb') as file:
                self.train = pickle.load(file)
                self.test = pickle.load(file)
        except Exception as ex:
            print("Error in loadData: %s" % str(ex))
            self._loadData_Keras()
            with open(self.preparedDatasetFileName, 'wb') as file:
                pickle.dump(self.train, file)
                pickle.dump(self.test, file)

        # from tensorflow.examples.tutorials.mnist import input_data
        #
        # mnist = input_data.read_data_sets('Data/Mnist', one_hot=True)

    def _loadData_Keras(self):
        from tensorflow.keras.datasets import mnist

        self.train = CMnistDataset.TSubset()
        self.test  = CMnistDataset.TSubset()
        (self.train.images, self.train.labels), \
                (self.test.images, self.test.labels) = mnist.load_data()
        self.train.images = self.train.images / np.float32(255.0)
        self.test.images  = self.test.images  / np.float32(255.0)
