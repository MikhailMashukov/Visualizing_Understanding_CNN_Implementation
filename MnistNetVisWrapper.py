# My TensorFlow\TensorWatch\mnist_watch.py, cut and attached to VisQtMain
from __future__ import absolute_import, division, print_function, unicode_literals

# import argparse
import datetime
import math
# import subprocess
import os
# import time

import numpy as np
# import psutil

import DataCache
from MyUtils import *

# FLAGS = None

class CBaseLearningCallback:   # Predeclaration. Without it doLearning(..., callback=CBaseLearningCallback() doesn't see this class
    pass

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

def getGradientTensors(model):
    """Return the gradient of every trainable weight in model
    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights."""
    # print('tensors')
    weights = []
    for tensor in model.trainable_weights:
        layerInfo = tensor.name.split('/')
        layerName = layerInfo[0]
        if model.get_layer(layerName).trainable:
            weights.append(tensor)
    optimizer = model.optimizer
    gradientTensors = optimizer.get_gradients(model.total_loss, weights)
    return gradientTensors


class CMnistVisWrapper:
    def __init__(self, mnistDataset=None, activationCache=None):
        self.name = 'mnist'
        self.weightsFileNameTempl = 'QtLogs/MnistWeights_Epoch%d.h5'
        self.gradientsFileNameTempl = 'QtLogs/MnistGrads_Epoch%d.dat'
        self.mnistDataset = CMnistDataset() if mnistDataset is None else mnistDataset
        self.net = None
        self.curEpochNum = 0
        self.isLearning = False
        self.curModelLearnRate = None
        self.cancelling = False
        self.activationCache = DataCache.CDataCache(256 * getCpuCoreCount()) \
                if activationCache is None else activationCache
        self.netsCache = None
        self.gradientTensors = None
        self.gradientKerasFunc = None

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

        # Converting to channels first, as VisQtMain expects (batch, channels, y, x)
        if len(activations.shape) == 4:
            activations = activations.transpose((0, -1, 1, 2))
        elif len(activations.shape) == 3:
            activations = activations.transpose((0, -1, 1))
        self.activationCache.saveObject(itemCacheName, activations)
        return activations

    def getImagesActivations_Batch(self, layerName, imageNums, epochNum=None):
        if epochNum is None:
            epochNum = self.curEpochNum

        batchActs = [None] * len(imageNums)
        images = []
        for i in range(len(imageNums)):
            imageNum = imageNums[i]
            itemCacheName = 'act_%s_%d_%d' % (layerName, imageNum, epochNum)
            cacheItem = self.activationCache.getObject(itemCacheName)
            if not cacheItem is None:
                batchActs[i] = cacheItem[0]
            else:
                imageData = self.mnistDataset.getImage(imageNum)
                images.append(imageData)
                # print('no data for ', itemCacheName)
        if not images:
            # print("Activations for batch taken from cache")
            return np.stack(batchActs, axis=0)

        model = self._getNet(layerName)
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
        imageData = np.stack(images, axis=0)
        print("Predict data prepared")
        activations = model.model.predict(imageData)   # np.expand_dims(imageData, 0), 3))
        print("Predicted")

        # Converting to channels first, as VisQtMain expects (batch, channels, y, x)
        if len(activations.shape) == 4:
            activations = activations.transpose((0, -1, 1, 2))
        elif len(activations.shape) == 3:
            activations = activations.transpose((0, -1, 1))

        predictedI = 0
        for i in range(len(imageNums)):
            if batchActs[i] is None:
                batchActs[i] = activations[predictedI]
                imageNum = imageNums[i]
                itemCacheName = 'act_%s_%d_%d' % (layerName, imageNum, epochNum)
                self.activationCache.saveObject(itemCacheName, batchActs[i : i + 1])
                predictedI += 1
        assert predictedI == activations.shape[0]
        if len(images) == len(imageNums):
            return activations
        print('Output prepared')
        return np.stack(batchActs, axis=0)

    # Returns convolutions' multiplication weights (not bias) or similar for other layers
    def getMultWeights(self, layerName):  # , epochNum):
        model = self._getNet()
        layer = model.base_model.get_layer(layerName)
        allWeights = layer.get_weights()
        assert len(allWeights) == 1 or len(allWeights[0].shape) > len(allWeights[1].shape)
        weights = allWeights[0]
        if len(weights.shape) == 4:
            weights = weights.transpose((2, 3, 0, 1))
        elif len(weights.shape) == 3:
            weights = weights.transpose((2, 0, 1))       # I suppose channels_first mean this
        return weights

        # for trainWeights in layer._trainable_weights:
        #     if trainWeights.name.find('bias') < 0:
        #         return trainWeights.numpy()
        # raise Exception('No weights found for layer %s' % layerName)

    def setMultWeights(self, layerName, weights):  # , epochNum):
        model = self._getNet()
        layer = model.base_model.get_layer(layerName)
        allWeights = layer.get_weights()
        if len(weights.shape) == 4:
            weights = weights.transpose((2, 3, 0, 1))
        elif len(weights.shape) == 3:
            weights = weights.transpose((1, 2, 0))
        allWeights[0] = weights
        layer.set_weights(allWeights)

        # for index, trainWeights in enumerate(allWeights):
        #     if trainWeights.name.find('bias') < 0:
        #         allWeights[index] = matWeights
        #         layer.set_weights(allWeights)
        #         return
        # raise Exception('No weights found for layer %s' % layerName)

    def getGradients(self, layerName, firstImageCount, epochNum=None):
        import tensorflow as tf
        import keras.backend as K

        if epochNum is None:
            epochNum = self.curEpochNum

        model = self._getNet()
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
            print("State loaded")
        if self.gradientTensors is None:
            self.gradientTensors = getGradientTensors(self.net.model)
            self.gradientKerasFunc = K.function(inputs=[model.base_model.input,
                       model.base_model._feed_sample_weights[0],
                       model.base_model.targets[0]],
                    outputs=self.gradientTensors)
        data = self.mnistDataset.getNetSource()
        inp = [data[0][:firstImageCount], 1, \
                   tf.keras.utils.to_categorical(data[1][:firstImageCount], num_classes=10)]
        print("Data for net prepared")
        gradients = self.gradientKerasFunc(inp)
        print("Gradients calculated")
        layerInd = None
        for i in range(len(self.gradientTensors)):
            name = self.gradientTensors[i].name
            if name.find(layerName) >= 0 and name.find('Bias') < 0:
                if layerInd != None:
                    raise Exception('Multiple matching gradients layers (%s and %s)' % \
                                    (self.gradientTensors[layerInd].name, name))
                layerInd = i
        if layerInd is None:
            raise Exception('Unknown layer %s' % layerName)
        gradients = gradients[layerInd]
        if len(gradients.shape) == 4:
            gradients = gradients.transpose((2, 3, 0, 1))    # Becomes (in channel, out channel, y, x)
        elif len(gradients.shape) == 3:
            gradients = gradients.transpose((-1, 0, 1))
        return gradients

        # gradientDict = dict()        # For saving to file
        # for i in range(len(gradientTensors)):
        #     gradientDict[gradientTensors[i].name] = gradients[i]
        # return gradientDict


    # Some epochs that will be run can be cut, not on entire dataset
    def doLearning(self, iterCount, options, callback=CBaseLearningCallback()):
        self.cancelling = False
        if self.net is None:
            self._initMainNet()
        self.isLearning = True
        try:
            epochNum = 0    # Number for starting from small epochs each time
            for _ in range(int(math.ceil(iterCount / 100))):
                if 0:
                    if iterCount > 500:
                        if epochNum < 4:
                            (start, end) = (epochNum * 1000, (epochNum + 1) * 1000)
                        elif 4 << (epochNum - 4) <= 55:
                            (start, end) = (2000 + 2000 << (epochNum - 4), 2000 + 4000 << (epochNum - 4))
                        else:
                            (start, end) = (0, None)
                    else:
                        (start, end) = (0, None)
                else:
                    shift = 2000 * (epochNum % 30)
                    (start, end) = (shift, shift + 2000)


                if self.curModelLearnRate != options.learnRate:
                    self.setLearnRate(options.learnRate)
                    print('Learning rate switched to %f' % options.learnRate)

                infoStr = self.net.doLearning(1, callback,
                                              start, end, self.curEpochNum)
                self.curEpochNum += 1
                epochNum += 1
                infoStr = 'Epoch %d: %s' % (self.curEpochNum, infoStr)
                self.saveState()
                # self.saveCurGradients()
                callback.onEpochEnd(self.curEpochNum, infoStr)
                if self.cancelling:
                    break
        finally:
            self.isLearning = False

        # self.activationCache.clear()
        return infoStr


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

    def saveCurGradients(self):
        import pickle

        with open(self.gradientsFileNameTempl % self.curEpochNum, 'wb') as file:
            gradientDict = self.getGradients()
            for name, value in gradientDict.items():
                pickle.dump(name, file)
                pickle.dump(value, file)

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
            if self.isLearning:
                raise Exception('Learning is in progress')
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
            return CMnistVisWrapper.get_conv_3_source_block
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
        self.setLearnRate(0.1)
        # if os.path.exists(self.weightsFileNameTempl):
        #     self.net.model.load_weights(self.weightsFileNameTempl)
        # self.net.model._make_predict_function()
        self.net.batchSize = min(16 * getCpuCoreCount(), 64)

        self.netsCache = dict()

    def setLearnRate(self, learnRate):
        from keras.optimizers import Adam, SGD

        # optimizer = Adam(learning_rate=learnRate, decay=1e-5)
        optimizer = SGD(lr=learnRate, decay=1e-6, momentum=0.9, nesterov=True)
        self.net.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.curModelLearnRate = learnRate

    def _getNet(self, highestLayer = None):
        if not self.net:
            self._initMainNet()

        if highestLayer is None:
            return self.net
        else:
            if not highestLayer in self.netsCache:
                import MnistNet

                self.netsCache[highestLayer] = MnistNet.CMnistRecognitionNet2(highestLayer, base_model=self.net.model)
                # self.netsCache[highestLayer].model._make_predict_function()
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


class CBaseLearningCallback:
    def __init__(self):
        self.lastUpdateTime = datetime.datetime.now()

    # Logs from CSummaryWriteCallback.on_batch_end and approximate iteration number
    def onBatchEnd(self, trainIterNum, logs):
        t = datetime.datetime.now()
        if (t - self.lastUpdateTime).total_seconds() >= 1:
            self.onSecondPassed()
            self.lastUpdateTime = t

    # Less frequent update callback
    def onSecondPassed(self):
        pass

    def onEpochEnd(self, curEpochNum, infoStr):
        pass

