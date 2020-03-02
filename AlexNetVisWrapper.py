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
from VisUtils import *

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

    def getTowerCount(self):
        return 4       # Actually 2, 2 * 2 - in order to have what to compare


    def getImageActivations(self, layer, imageNum, _): # epochNum=None):
        # epochNum is not used here
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

    def getImagesActivations_Batch(self, layer, imageNums, epochNum=None):
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

    def getMultWeights(self, layerName, allowCombinedLayers=False):
        itemCacheName = 'w_%s' % (layerName)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        model = self._getAlexNet().model
        try:
            layer = model.get_layer(layerName)
            allWeights = layer.get_weights()
        except:
            allWeights = None

        if allWeights:
            assert len(allWeights) == 1 or len(allWeights[0].shape) > len(allWeights[1].shape)
            weights = allWeights[0]
        else:
            if allowCombinedLayers:
                allWeights = []
                for layer in model.layers:
                    if layer.name.find(layerName + '_') == 0:
                        allLayerWeights = layer.get_weights()
                        assert len(allLayerWeights) == 1 or len(allLayerWeights[0].shape) > len(allLayerWeights[1].shape)
                        allWeights.append(allLayerWeights[0])
                if not allWeights:
                    raise Exception('No weights found for combined layer %s' % layerName)
                weights = np.concatenate(allWeights, axis=3)
            else:
                raise Exception('No weights found for layer %s' % layerName)

        # Converting to channels_last
        if len(weights.shape) == 4:
            weights = weights.transpose((2, 3, 0, 1))
        elif len(weights.shape) == 3:
            weights = weights.transpose((2, 0, 1))
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

    def calcWeightsVisualization2(self, layerName):
        allLayerNames = self.getNetLayersToVisualize()
        layersWeights = []
        for layerName2 in allLayerNames:
            layersWeights.append(self.getMultWeights(layerName2, True))  # E.g. [3, 96, 9, 9]
            if layerName2 == layerName:
                break

        curImageData = layersWeights[0]
        assert(len(curImageData.shape) == 4)
        curImageData = curImageData.transpose([1, 2, 3, 0])  # Channels, x, y (or y, x, not sure), colors
        layersStrides = [2, 1]
        for layerInd1, weights2 in enumerate(layersWeights[1:]):
            if 1:
                prev = curImageData
                curImageData = curImageData.transpose([0, 3, 1, 2])
                curImageData = pool2d(curImageData, kernel_size=3, stride=2, padding=0, pool_mode='max')
                # curImageData = np.array([pool2d(chan, kernel_size=3, stride=2, padding=2, pool_mode='max') \
                #                          for chan in curImageData])
                curImageData = curImageData.transpose([0, 2, 3, 1])

            strides = layersStrides[layerInd1]
            assert(len(weights2.shape) == 4)      # E.g. [96, 256, 3, 3]
            newImageData = []
            if curImageData.shape[0] != weights2.shape[0]:
                assert curImageData.shape[0] > weights2.shape[0] and \
                       curImageData.shape[0] % weights2.shape[0] == 0
                towerCount = curImageData.shape[0] // weights2.shape[0]
            else:
                towerCount = 1

            for outChanInd in range(weights2.shape[1]):
            # for outChanInd in range(5):  #d_
                curSum = np.zeros((curImageData.shape[1] + strides * (weights2.shape[2] - 1),
                                   curImageData.shape[2] + strides * (weights2.shape[3] - 1),
                                   curImageData.shape[3]))
                if towerCount == 1:
                    startInChanInd = 0
                else:
                    startInChanInd = outChanInd // (weights2.shape[1] // towerCount) * weights2.shape[0]
                for chanInd in range(weights2.shape[0]):
                            # i = 0
                            # j = 0
                    for i in range(weights2.shape[2]):
                        for j in range(weights2.shape[2]):
                            curSum[strides * i : strides * i + curImageData.shape[1],
                                   strides * j : strides * j + curImageData.shape[2], :] += \
                                    curImageData[startInChanInd + chanInd] * \
                                        (weights2[chanInd, outChanInd, i, j]) # - weights2[chanInd].min())
                newImageData.append(curSum)
            curImageData = np.stack(newImageData, axis=0)

        # if len(weights.shape) == 4:
        #     weights = weights.transpose([1, 2, 3, 0])  # Channels, x, y (or y, x, not sure), colors
        # else:
        #     raise Exception('Not supported weights\' shape')

        if 0:
            curImageData -= curImageData.min()
            curImageData = np.array(curImageData /
                    max([abs(curImageData.min()), abs(curImageData.max())]) * 255.0, dtype=np.uint8)
        else:
            print('Weights visualization min %.5f, max %.5f, std. dev. %.7f' % \
                  (curImageData.min(), curImageData.max(), np.std(curImageData)))
            div = np.std(curImageData) * 6
            if (curImageData.max() - curImageData.min()) / div < 1.2:
                curImageData -= curImageData.mean()
                curImageData = curImageData / div + 0.5
            else:
                curImageData -= curImageData.min()
                curImageData = curImageData / curImageData.max() * 1.2 - 0.1
            print('New min %.5f, max %.5f, std. dev. %.7f' % \
                  (curImageData.min(), curImageData.max(), np.std(curImageData)))
            curImageData[curImageData < 0] = 0
            curImageData[curImageData > 1] = 1
            curImageData = np.array(curImageData * 255.0, dtype=np.uint8)
        return curImageData

    def doLearning(self, iterCount, options, callback=None):
        self.cancelling = False
        if self.net is None:
            self._getAlexNet()
        raise Exception('Not implemented')


    def showProgress(self, str, processEvents=True):
        print(str)

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
        return [1]     # Not implemented yet

    # This network only saves caches for further faster starting up
    def saveState(self):
        try:
            with open('Data/VisImagesCache.dat', 'wb') as file:
                self.cache.saveState_OpenedFile(file)
            with open('Data/VisActivationsCache.dat', 'wb') as file:
                self.activationCache.saveState_OpenedFile(file)
        except Exception as ex:
            self.showProgress("Error in saveState: %s" % str(ex))

    def loadState(self, epocNum=None):
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
        try:
            import pandas
    
            #with open('Data/ILSVRC2012_classes.txt', mode='r'):
            self.testLabels = pandas.read_table('Data/ILSVRC2012_classes.txt', 
                    dtype=int, header=None, squeeze=True)
        except Exception as ex:
            print("Error on loading 'ILSVRC2012_classes.txt': %s" % str(ex))
            self.testLabels = np.ones([50000])

    def getImageFilePath(self, imageNum):
        return 'Images/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG' % imageNum

    def getImage(self, imageNum, preprocessStage='net'):
        itemCacheName = self._getImageCacheName(imageNum, preprocessStage)
        cacheItem = self.cache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        import alexnet_utils

        imgFileName = self.getImageFilePath(imageNum)

        if preprocessStage == 'source':
            imageData = alexnet_utils.imread(imgFileName, pilmode='RGB')
        elif  preprocessStage == 'cropped':   # Cropped and resized, as for alexnet
                # but in uint8, without normalization and transposing back and forth.
                # Float32 lead to incorrect colors in imshow
            img_size=(256, 256)
            crop_size=(227, 227)
            imageData = alexnet_utils.imread(imgFileName, pilmode='RGB')
            imageData = alexnet_utils.imresize(imageData, img_size)
            imageData = imageData[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2,
                (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2, :]
            # imageData[:, [1, 4, 7], :] = [[255, 0, 0], [0, 200 ,0], [0, 0, 145]]
        else:
            imageData = alexnet_utils.preprocess_image_batch([imgFileName])[0]
            imageData = imageData.transpose((1, 2, 0))

        self.cache.saveObject(itemCacheName, imageData)
        return imageData

    # This method suits ILSVRC's data poorly
    def getNetSource(self, type='train'):
        # if self.test is None:
        #     self.loadData()
        # subset = self.train if type == 'train' else self.test
        if type != 'test':
            print('Warning: no AlexNet train images')
        data = []
        for imageInd in range(1000):
            data.append(self.getImage(imageInd + 1, 'net'))
        return (np.stack(data, axis=0), self.testLabels)

    def _getImageCacheName(self, imageNum, preprocessStage):
        return 'im_%d_%s' % (imageNum, preprocessStage)

