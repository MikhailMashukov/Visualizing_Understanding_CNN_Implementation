# My TensorFlow\TensorWatch\mnist_watch.py, cut and attached to VisQtMain
from __future__ import absolute_import, division, print_function, unicode_literals

# import argparse
# import datetime
import math
# import subprocess
import os
# import time

import numpy as np
# import psutil

import DataCache
from MyUtils import *

# FLAGS = None

def getSavedNetEpochs(fileMask):     # E.g. QtLogs/MNISTWeights_epoch*.h5
    import glob
    import re

    epochNums = []
    # for fileName in os.listdir(fileMask):
    for fileName in glob.glob(fileMask):
        fileName = fileName.lower()
        result = re.search(r'epoch(\d+)\.', fileName)
        # p = fileName.find('epoch')
        if result:
            epochNums.append(int(result.group(1)))
    return sorted(epochNums)


class CMnistVisWrapper:
    def __init__(self):
        self.name = 'mnist'
        self.weightsFileNameTempl = 'QtLogs/MnistWeights_Epoch%d.h5'
        self.mnistDataset = CMnistDataset()
        self.net = None
        self.curEpochNum = 0
        self.activationCache = DataCache.CDataCache(64 * getCpuCoreCount())
        self.netsCache = None

    def getImageDataset(self):
        return self.mnistDataset

    def getNetLayersToVisualize(self):
        return ['conv_1', 'conv_2', 'conv_3', 'dense_1', 'dense_2']

    def getImageActivations(self, layerName, imageNum, epochNum=None):
        if epochNum is None:
            epochNum = self.curEpochNum
        itemCacheName = 'act_%s_%d_%d' % (layerName, imageNum, epochNum)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        model = self._getNet(layerName)
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
        imageData = self.mnistDataset.getImage(imageNum)
        imageData = np.expand_dims(imageData, 0)
        activations = model.model.predict(imageData)   # np.expand_dims(imageData, 0), 3))

        # Converting to channels first, as VisQtMain expects
        if len(activations.shape) == 4:
            activations = activations.transpose((0, -1, 1, 2))
        elif len(activations.shape) == 3:
            activations = activations.transpose((0, -1, 1))
        self.activationCache.saveObject(itemCacheName, activations)
        return activations

    def doLearning(self, iterCount):
        if self.net is None:
            self._initMainNet()
        for _ in range(int(math.ceil(iterCount / 100))):
            infoStr = self.net.doLearning(100)
            self.curEpochNum += 1
            self.saveState()
        # self.activationCache.clear()
        return 'Epoch %d: %s' % (self.curEpochNum, infoStr)


    def getSavedNetEpochs(self):
        return getSavedNetEpochs(self.weightsFileNameTempl.replace('%d', '*'))

    def saveState(self):
        try:
            with open('Data/MnistVisActCache.dat', 'wb') as file:
                self.activationCache.saveState_OpenedFile(file)
            if not self.net is None:
                self.net.model.save_weights(self.weightsFileNameTempl % self.curEpochNum)
        except Exception as ex:
            print("Error in saveState: %s" % str(ex))

    # When it is desirable to initialize quickly (without tensorflow)
    def loadCacheState(self):
        try:
            with open('Data/MnistVisActCache.dat', 'rb') as file:
                self.activationCache.loadState_OpenedFile(file)
        except Exception as ex:
            print("Error in loadCacheState: %s" % str(ex))

    def loadState(self, epochNum=-1):
        try:
            # self.loadCacheState()
            if self.net is None:
                self._initMainNet()
            self.net.model.load_weights(self.weightsFileNameTempl % epochNum)
            self.curEpochNum = epochNum
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

    def getCacheStatusInfo(self):
        return '%.2f MBs' % \
                (self.activationCache.getUsedMemory() / (1 << 20))

    @staticmethod
    def get_source_block_calc_func(layerName):
        if layerName == 'conv_1':
            return CMnistVisWrapper.get_conv_1_source_block
        elif layerName == 'conv_2':
            return CMnistVisWrapper.get_conv_2_source_block
        elif layerName == 'conv_3':
            return CMnistVisWrapper.get_conv_2_source_block
        else:
            return CMnistVisWrapper.get_entire_image_block

    # Returns source pixels block, corresponding to the layer conv_1 pixel (x, y)
    @staticmethod
    def get_conv_1_source_block(x, y):
        source_xy_0 = (x * 2, y * 2)
        size = 5
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_2_source_block(x, y):
        source_xy_0 = (x * 2, y * 2)
        size = 9
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_3_source_block(x, y):
        source_xy_0 = (x * 4, y * 4)
        size = 11 + 4 * 2
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_entire_image_block(_, y):
        return (0, 0, 28, 28)


    def _initMainNet(self):
        import MnistNet

        self.net = MnistNet.CMnistRecognitionNet2()
        dataset = CMnistDataset()
        self.net.init(dataset, 'QtLogs')
        # if os.path.exists(self.weightsFileNameTempl):
        #     self.net.model.load_weights(self.weightsFileNameTempl)

        self.netsCache = dict()

    def _getNet(self, highestLayer = None):
        if not self.net:
            self._initMainNet()

        if highestLayer is None:
            return self.net
        else:
            if not highestLayer in self.netsCache:
                import MnistNet

                self.netsCache[highestLayer] = MnistNet.CMnistRecognitionNet2(highestLayer, base_model=self.net.model)
            return self.netsCache[highestLayer]


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
                # self.train.images = self.train.images[:5000]
                # self.train.labels = self.train.labels[:5000]
                self.test = pickle.load(file)
        except Exception as ex:
            print("Error in CMnistDataset.loadData: %s" % str(ex))
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
