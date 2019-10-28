# import copy
# import datetime
import pickle
# import math
import numpy as np
import os
# import random
import sys
import time

import DataCache
from MyUtils import *

class CAlexNetVisWrapper:
    def __init__(self):
        self.name = 'alexnet'
        self.net = None
        self.cache = DataCache.CDataCache(256 * getCpuCoreCount())
        self.activationCache = DataCache.CDataCache(64 * getCpuCoreCount())
        self.imageDataset = CImageDataset(self.cache)

        self.isLearning = False
        self.netsCache = None

    def getImageDataset(self):
        return self.imageDataset

    def getNetLayersToVisualize(self):
        layerNames = []
        for i in range(5):
            layerNames.append('conv_%d' % (i + 1))
        for i in range(2):
            layerNames.append('dense_%d' % (i + 1))
        return layerNames


    def getImageActivations(self, layer, imageNum):
        if isinstance(layer, int):
            layerName = 'conv_%d' % layer
        else:
            layerName = layer
        itemCacheName = 'act_%s_%d' % (layerName, imageNum)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        model = self._getAlexNet(layerName)
        imageData = self.imageDataset.getImage(imageNum, 'net')
        imageData = imageData.transpose((2, 0, 1))
        activations = model.model.predict(np.expand_dims(imageData, 0))              # About 30 ms
        # activations = model.predict(self.imageDataset.getImageFilePath(imageNum))  # About 50 ms
        self.activationCache.saveObject(itemCacheName, activations)
        return activations

    def getImagesActivations_Batch(self, layer, imageNums):
        if isinstance(layer, int):
            layerName = 'conv_%d' % layer
        else:
            layerName = layer

        imageDataList = []
        for imageNum in imageNums:
            imageData = self.imageDataset.getImage(imageNum, 'net')
            imageDataList.append(imageData.transpose((2, 0, 1)))
        batchInput = np.stack(imageDataList)

        model = self._getAlexNet(layerName)
        activations = model.model.predict(batchInput)
        return activations

    def getMultWeights(self, layerName):
        itemCacheName = 'w_%s' % (layerName)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        model = self._getAlexNet().model
        if layerName == 'conv_3':
            allWeights = model.layers[13]._trainable_weights
        else:
            raise Exception('Unknown weights position in net')
        weights = allWeights[0].numpy()
        self.activationCache.saveObject(itemCacheName, weights)
        return weights

    def getImagesActivationMatrix(self, layerNum, mode='summation'):
        itemCacheName = 'actMat_%s_%d' % (mode, layerNum)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        import pandas

        activation_matrix_filename = 'Data/Activations_{}/Strongest_Activation_Layer{}.csv'.format(
                mode, layerNum)
        with open(activation_matrix_filename, mode='r'):
            activations = pandas.read_table(activation_matrix_filename, dtype=np.float32, header=None).as_matrix()
        checkVals = [np.min(activations[0, :]), np.max(activations[0, :]), \
                     np.min(activations[:, 0]), np.max(activations[:, 0])]
        for checkVal in checkVals:
            if checkVal != 0:
                raise Exception('Unexpected non-zero value (%f) in %s' % \
                                (checkVal, activation_matrix_filename))
        activations = activations[1:, 1:]
        self.activationCache.saveObject(itemCacheName, activations)
        # self.saveState()
        return activations

    def doLearning(self, iterCount, options, callback=None):
        self.cancelling = False
        if self.net is None:
            self._getAlexNet()
        raise Exception('Not implemented')


    def getCacheStatusInfo(self):
        return '%.2f + %.2f MBs' % \
                (self.cache.getUsedMemory() / (1 << 20), \
                 self.activationCache.getUsedMemory() / (1 << 20))

    @staticmethod
    def get_source_block_calc_func(layerName):
        if layerName == 'conv_1':
            return CAlexNetVisWrapper.get_conv_1_source_block
        elif layerName == 'conv_2':
            return CAlexNetVisWrapper.get_conv_2_source_block
        elif layerName == 'conv_3':
            return CAlexNetVisWrapper.get_conv_3_source_block
        elif layerName == 'conv_4':
            return CAlexNetVisWrapper.get_conv_4_source_block
        elif layerName == 'conv_5':
            return CAlexNetVisWrapper.get_conv_5_source_block
        elif layerName[:6] == 'dense_':
            return CAlexNetVisWrapper.get_entire_image_block
        else:
            return None

    # Returns source pixels block, corresponding to the layer conv_1 pixel (x, y)
    @staticmethod
    def get_conv_1_source_block(x, y):
        source_xy_0 = (x * 4, y * 4)
        size = 11
        return (source_xy_0[0], source_xy_0[1], source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_2_source_block(x, y):
        source_xy_0 = ((x - 2) * 8, (y - 2) * 8)
        size = 51  # 11 + 4 * 2 + 8 * 4
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_3_source_block(x, y):
        source_xy_0 = ((x - 2) * 16, (y - 2) * 16)
        size = 99  # 51 + 8 * 2 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_4_source_block(x, y):
        source_xy_0 = ((x - 3) * 16, (y - 3) * 16)
        size = 131  # 99 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_conv_5_source_block(x, y):
        source_xy_0 = ((x - 4) * 16, (y - 4) * 16)
        size = 163  # 131 + 16 * 2
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + size, source_xy_0[1] + size)

    @staticmethod
    def get_entire_image_block(x, y):
        return (0, 0, 227, 227)


    def getRecommendedLearnRate(self):
        # return 0.1      # SGD
        return 0.001    # Adam

    def _getAlexNet(self, highestLayer = None):
        if not self.net:
            import alexnet

            self.net = alexnet.AlexNet()
            self.netsCache = dict()

        if highestLayer is None:
            return self.net
        else:
            if not highestLayer in self.netsCache:
                import alexnet

                self.netsCache[highestLayer] = alexnet.AlexNet(highestLayer, base_model=self.net.model)
            return self.netsCache[highestLayer]

    def getSavedNetEpochs(self):
        return []     # Not implemented yet

    # This network only saves caches for further faster starting up
    def saveState(self):
        try:
            with open('Data/VisImagesCache.dat', 'wb') as file:
                self.cache.saveState_OpenedFile(file)
            with open('Data/VisActivationsCache.dat', 'wb') as file:
                self.activationCache.saveState_OpenedFile(file)
        except Exception as ex:
            self.showProgress("Error in saveState: %s" % str(ex))

    def loadState(self):
        try:
            with open('Data/VisImagesCache.dat', 'rb') as file:
                self.cache.loadState_OpenedFile(file)
            with open('Data/VisActivationsCache.dat', 'rb') as file:
                self.activationCache.loadState_OpenedFile(file)
        except Exception as ex:
            self.showProgress("Error in loadState: %s" % str(ex))


class CImageDataset:
    def __init__(self, cache):
        self.cache = cache

    def getImageFilePath(self, imageNum):
        return 'ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG' % imageNum

    def getImage(self, imageNum, preprocessStage='net'):
        itemCacheName = self._getImageCacheName(imageNum, preprocessStage)
        cacheItem = self.cache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        import alexnet_utils

        imgFileName = self.getImageFilePath(imageNum)

        if preprocessStage == 'source':
            imageData = alexnet_utils.imread(imgFileName, mode='RGB')
        elif  preprocessStage == 'cropped':   # Cropped and resized, as for alexnet
                # but in uint8, without normalization and transposing back and forth.
                # Float32 lead to incorrect colors in imshow
            img_size=(256, 256)
            crop_size=(227, 227)
            imageData = alexnet_utils.imread(imgFileName, mode='RGB')
            imageData = alexnet_utils.imresize(imageData, img_size)
            imageData = imageData[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2,
                (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2, :]
            # imageData[:, [1, 4, 7], :] = [[255, 0, 0], [0, 200 ,0], [0, 0, 145]]
        else:
            imageData = alexnet_utils.preprocess_image_batch([imgFileName])[0]
            imageData = imageData.transpose((1, 2, 0))

        self.cache.saveObject(itemCacheName, imageData)
        return imageData

    def _getImageCacheName(self, imageNum, preprocessStage):
        return 'im_%d_%s' % (imageNum, preprocessStage)

