from __future__ import absolute_import, division, print_function, unicode_literals

# import datetime
import math
# import random
import os
import psutil
# import time
import numpy as np
import warnings

import DeepOptions
import DataCache
from MyUtils import *
from MnistNetVisWrapper import CBaseLearningCallback, getSavedNetEpochs
# import AlexNetVisWrapper
from ImageNetsVisWrappers import CImageNetPartDataset, CSourceBlockCalculator

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class CPyTorchImageNetVisWrapper:
    def __init__(self, imageDataset=None, activationCache=None):
        self.name = 'image'
        self.weightsFileNameTempl = 'PyTLogs/checkpoints/PyTImWeights_Epoch%d.pkl'
        # self.weightsFileNameTempl = 'PyTLogs/checkpoints/PyTImWeights_Epoch%d.h5'
        self.imageCache = DataCache.CDataCache(256 * getCpuCoreCount())
        self.imageDataset = CImageNetPartDataset(self.imageCache) if imageDataset is None else imageDataset
        # self.imageDataset = CAugmentedMnistDataset(CImageNetPartDataset()) if imageDataset is None else imageDataset
        self.net = None
        self.optimizer = None
        self.netPreprocessStageName = 'net'  # for self.net = alexnet.AlexNet()
        self.doubleSizeLayerNames = []    # Not stored in saveState
        self.curEpochNum = 0
        self.isLearning = False
        self.curModelLearnRate = None
        self.cancelling = False
        self.activationCache = DataCache.CDataCache(128 * getCpuCoreCount()) \
                if activationCache is None else activationCache
        self.netsCache = None
        self.gradientTensors = None
        self.gradientKerasFunc = None

    def getImageDataset(self):
        return self.imageDataset

    def getNetLayersToVisualize(self):
        return ['conv_1', 'conv_2', 'conv_3'] + \
            ['conv_11', 'conv_12', 'conv_13', 'conv_21', 'conv_22', 'conv_23', 'max_pool_22', 'concat_23',
             'conv_24', 'conv_25',
             'dense_1', 'dense_2', 'dense_3']

    def getComponentNetLayers(self):
        return self.getNetLayersToVisualize()

    def getTowerCount(self):
        return DeepOptions.towerCount

    def getImageActivations(self, layerName, imageNum, epochNum=None):
        if epochNum is None or epochNum < 0:
            epochNum = self.curEpochNum
        itemCacheName = 'act_%s_%d_%d' % (layerName, imageNum, epochNum)
        cacheItem = self.activationCache.getObject(itemCacheName)
        if not cacheItem is None:
            return cacheItem

        model = self._getNet(layerName)
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
        imageData = self.imageDataset.getImage(imageNum, self.netPreprocessStageName)
        imageData = np.expand_dims(imageData, 0)
        activations = model.predict(imageData)   # np.expand_dims(imageData, 0), 3))

        if self.netPreprocessStageName == 'net':
            activations = self._transposeToOutBatchDims(activations)
        self.activationCache.saveObject(itemCacheName, activations)
        return activations

    def getImagesActivations_Batch(self, layerName, imageNums, epochNum=None):
        if epochNum is None or epochNum < 0:
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
                imageData = self.imageDataset.getImage(imageNum, self.netPreprocessStageName, 'train')
                images.append(imageData)
                # print('no data for ', itemCacheName)
        if not images:
            # print("Activations for batch taken from cache")
            return np.stack(batchActs, axis=0)

        model = self._getNet(layerName)
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
        imageData = np.stack(images, axis=0)
        # print("Predict data prepared")
        activations = model.model.predict(imageData)   # np.expand_dims(imageData, 0), 3))
        # print("Predicted")

        if self.netPreprocessStageName == 'net':
            activations = self._transposeToOutBatchDims(activations)

        predictedI = 0
        for i in range(len(imageNums)):
            if batchActs[i] is None:
                batchActs[i] = activations[predictedI]
                imageNum = imageNums[i]
                itemCacheName = 'act_%s_%d_%d' % (layerName, imageNum, epochNum)
                self.activationCache.saveObject(itemCacheName, np.expand_dims(batchActs[i], 0))
                predictedI += 1
        assert predictedI == activations.shape[0]
        if len(images) == len(imageNums):
            return activations
        print('Output prepared')
        return np.stack(batchActs, axis=0)

    # Returns convolutions' multiplication weights (not bias) or similar for other layers.
    # AllowCombinedLayers == True means than internal layers with names like 'conv_2_0' and 'conv_2_1'
    # can be matched to specified 'conv_2' and theirs weights transparently united into one tensor
    def getMultWeights(self, layerName, allowCombinedLayers=False):  # , epochNum):
        model = self._getNet()

        try:
            layer = model.getLayer(layerName)
            allWeights = layer.state_dict()
        except:
            allWeights = None
#         print('len ', len(allWeights))

        if allWeights:
#             assert len(allWeights) == 1 or 'weight' in allWeights
            weights = allWeights['weight']
        else:
            if allowCombinedLayers:
                allWeights = []
                for curLayerName, layer in model.getAllLayers():
                    if curLayerName.find(layerName + '_') == 0:
                        allLayerWeights = layer.get_weights()
#                         assert len(allLayerWeights) == 1 or len(allLayerWeights[0].shape) > len(allLayerWeights[1].shape)
                        allWeights.append(allLayerWeights['weight'])
                if not allWeights:
                    raise Exception('No weights found for combined layer %s' % layerName)
                weights = np.concatenate(allWeights, axis=3)
            else:
                raise Exception('No weights found for layer %s' % layerName)

        weights = weights.numpy()      # E.g. [96, 3, 11, 11]
        # Converting to channels_last
#         print(weights.shape)
        if len(weights.shape) == 5:
            weights = weights.transpose((2, 3, 4, 0, 1))    # Not tested
        elif len(weights.shape) == 4:
            weights = weights.transpose((1, 0, 2, 3))
        elif len(weights.shape) == 3:
            weights = weights.transpose((2, 0, 1))          # Not tested
        return weights

        # for trainWeights in layer._trainable_weights:
        #     if trainWeights.name.find('bias') < 0:
        #         return trainWeights.numpy()
        # raise Exception('No weights found for layer %s' % layerName)

    def setMultWeights(self, layerName, weights):  # , epochNum):
        model = self._getNet()
        layer = model.base_model.get_layer(layerName)
        allWeights = layer.get_weights()
        if len(weights.shape) == 5:
            weights = weights.transpose((2, 3, 4, 0, 1))   # Not sure here
        elif len(weights.shape) == 4:
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

    def getVariableValue(self, varName):
        import keras.backend as K

        model = self._getNet()
        var = model.base_model.variables['towers_weights']
        return var.numpy()

    def setVariableValue(self, varName, value):
        import keras.backend as K

        model = self._getNet()
        K.set_value(model.base_model.variables['towers_weights'], value)


    def getGradients(self, layerName, startImageNum, imageCount, epochNum=None, allowCombinedLayers=True):
        import tensorflow as tf
        import keras.backend as K

        if epochNum is None or epochNum < 0:
            epochNum = self.curEpochNum

        model = self._getNet()
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
            print("State loaded")
        if self.gradientTensors is None:
            self.gradientTensors = getGradientTensors(self.net)
            self.gradientKerasFunc = K.function(inputs=[model.base_model.input,
                       model.base_model._feed_sample_weights[0],
                       model.base_model.targets[0]],
                    outputs=self.gradientTensors)
        data = self.imageDataset.getNetSource()
        inp = [data[0][startImageNum : startImageNum + imageCount], 1, \
                   tf.keras.utils.to_categorical(data[1][startImageNum : startImageNum + imageCount], num_classes=10)]
        print("Data for net prepared")
        gradients = self.gradientKerasFunc(inp)
        print("Gradients calculated")

        # Looking for exact layer name match
        layerInd = None
        for i in range(len(self.gradientTensors)):
            name = self.gradientTensors[i].name
            if name.find(layerName + '/') >= 0 and name.find('Bias') < 0:
                if layerInd != None:
                    raise Exception('Multiple matching gradients layers (%s and %s)' % \
                                    (self.gradientTensors[layerInd].name, name))
                layerInd = i
        if layerInd is None:
            if allowCombinedLayers:
                # Gathering <layerName>_* layers instead
                selectedGradients = []
                names=[]
                for i in range(len(self.gradientTensors)):
                    name = self.gradientTensors[i].name
                    if name.find(layerName + '_') >= 0 and name.find('Bias') < 0:
                        selectedGradients.append(gradients[i])
                        names.append(name)
                # print('Returning conbined gradients for %s instead of %s' % (str(names), layerName))
                gradients = np.concatenate(selectedGradients, axis=3)
            else:
                raise Exception('Unknown layer %s' % layerName)
        else:
            gradients = gradients[layerInd]

        if len(gradients.shape) == 5:
            gradients = gradients.transpose((2, 3, 4, 0, 1)) # Not sure here
        elif len(gradients.shape) == 4:
            gradients = gradients.transpose((2, 3, 0, 1))    # Becomes (in channel, out channel, y, x)
        elif len(gradients.shape) == 3:
            gradients = gradients.transpose((-1, 0, 1))
        return gradients

        # gradientDict = dict()        # For saving to file
        # for i in range(len(gradientTensors)):
        #     gradientDict[gradientTensors[i].name] = gradients[i]
        # return gradientDict

    # Similar to CAlexNetVisWrapper.calcWeightsVisualization2. This approach has proven to be quite useless,
    # so here there is a cut version, working only for the first convolution layer
    def calcWeightsVisualization2(self, layerName):
        curImageData = self.getMultWeights(layerName, True)  # E.g. [3, 96, 9, 9]
        assert(len(curImageData.shape) == 4)
        curImageData = curImageData.transpose([1, 2, 3, 0])  # Channels, x, y (or y, x, not sure), colors

        if 1:
            print('Weights visualization %s min %.5f, max %.5f, std. dev. %.7f' % \
                  (str(curImageData.shape), curImageData.min(), curImageData.max(), np.std(curImageData)))
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


    # Some epochs that will be run can be cut, not on entire dataset
    def doLearning(self, iterCount, options, callback=CBaseLearningCallback()):
        self.cancelling = False
        if self.net is None:
            self._initMainNet()
        self.isLearning = True
        try:
            epochNum = 0    # Number for starting from small epochs each time

            trainImageNums = np.arange(1, self.imageDataset.getImageCount('train') + 1)
            testImageNums = np.arange(1, self.imageDataset.getImageCount('test') + 1)
                # TODO: actually doesn't pass to image dataset now

            # fullTestDataset = self.imageDataset.getNetSource('test')
            if 1:
            #     # Old variant, without augmentation but with simple CimageDataset support
            #
            #     fullDataset = self.imageDataset.getNetSource('train')
                if not options.trainImageNums is None:
                    trainImageNums = options.trainImageNums
                    if options.additTrainImageCount > 0:
                        trainImageNums = np.concatenate([trainImageNums, np.random.randint(
                                low=1, high=self.imageDataset.getImageCount('train'),
                                size=options.additTrainImageCount)])
            #         trainDataset = (fullDataset[0][imageNums - 1],
            #                        fullDataset[1][imageNums - 1])
            # else:
            #     # Optionally augmented and optimized for this variant
            #
            #     if options.trainImageNums is None:
            #         trainDataset = self.imageDataset.getAugmentedImages(epochImageCount)
            #     else:
            #         imageNums = options.trainImageNums
            #         if options.additTrainImageCount > 0:
            #             fullDataset = self.imageDataset.getNetSource('train')
            #             imageNums = np.concatenate([imageNums, np.random.randint(
            #                     low=1, high=fullDataset[0].shape[0], size=options.additTrainImageCount)])
            #         trainDataset = self.imageDataset.getAugmentedImagesForNums(imageNums)

            for _ in range(int(math.ceil(iterCount / 100))):
                self.printProgress('Learn. rate %.3g, batch %d' % \
                        (options.learnRate, self.net.batchSize))
                if 1:
                    if iterCount > 500:
                        if epochNum < 4:
                            epochImageCount = 1000
                        elif 2 << (epochNum - 4) <= 10:
                            epochImageCount = 2000 << (epochNum - 4)
                        else:
                            epochImageCount = 10000
                    else:
                        epochImageCount = 2000
                else:
                    epochImageCount = 2000
                if DeepOptions.imagesMainFolder.lower().find('imagenet') >= 0 and \
                   DeepOptions.imagesMainFolder != 'ImageNetPart':
                    epochImageCount *= 2

                if self.curModelLearnRate != options.learnRate:
                    self.setLearnRate(options.learnRate)
                    print('Learning rate switched to %f' % options.learnRate)

                # print('Train images: ', trainImageNums)
                if epochNum == 0:
                    infoStr = self.net.doLearning(1, callback,
                            trainImageNums, testImageNums,
                            epochImageCount, self.curEpochNum)
                else:
                    infoStr = self.net.doLearningWithPrevDataset(1, callback,
                            epochImageCount, self.curEpochNum)
                self.curEpochNum += 1
                epochNum += 1
                infoStr = 'Epoch %d: %s' % (self.curEpochNum, infoStr)
                if epochNum < 10 or epochNum % 10 == 0:
                    print(self.getCacheStatusInfo(True))
                self.printProgress(infoStr)

                self.saveState(self.curEpochNum % 8 == 0)
                # self.saveCurGradients()

                callback.onEpochEnd(self.curEpochNum, infoStr)
                if self.cancelling:
                    break
                if psutil.swap_memory().used > 16 << 30:
                    raise Exception('Stopping because of too high swap file usage')
        finally:
            self.isLearning = False

        # self.activationCache.clear()
        return infoStr


    def getSavedNetEpochs(self):
        return getSavedNetEpochs(self.weightsFileNameTempl.replace('%d', '*'))

    def saveState(self, saveCache=True):
        if saveCache:
            self.saveCacheState()
        try:
            if not self.net is None:
                self.net.model.save_weights(self.weightsFileNameTempl % self.curEpochNum)
        except Exception as ex:
            print("Error in saveState: %s" % str(ex))

    def saveCacheState(self):
        try:
            with open('Data/PyTNetVisActCache.dat', 'wb') as file:
                self.activationCache.saveState_OpenedFile(file)
        except Exception as ex:
            print("Error in saveState: %s" % str(ex))

    # When it is desirable to initialize quickly (without tensorflow)
    def loadCacheState(self):
        try:
            with open('Data/PyTNetVisActCache.dat', 'rb') as file:
                self.activationCache.loadState_OpenedFile(file)
        except Exception as ex:
            print("Error in loadCacheState: %s" % str(ex))

    def loadState(self, epochNum=-1):
        # try:
            # self.loadCacheState()
            if self.net is None:
                self._initMainNet()
            if self.isLearning:
                raise Exception('Learning is in progress')

            # if hasattr(self.net.model, 'weightsModel'):
            #     try:
            #         self.net.model.weightsModel.load_weights(self.weightsFileNameTempl % epochNum)
            #         print("Loaded weights for other model")
            #     except Exception as ex:
            #         print("Error on loading weights to other model: %s" % str(ex))
            #         self.net.model.load_weights(self.weightsFileNameTempl % epochNum)
            # else:
            self.net.loadState(self.weightsFileNameTempl % epochNum)
            self.curEpochNum = epochNum
        # except Exception as ex:
        #     print("Error in loadState: %s" % str(ex))

    def getCacheStatusInfo(self, detailed=False):
        str = '%.2f MBs' % \
                (self.activationCache.getUsedMemory() / (1 << 20))
        if detailed:
            cache = self.imageCache
            str = 'acts: %s, images: %s, %.2f MBs' % \
                    (str, cache.getDetailedUsageInfo(), cache.getUsedMemory() / (1 << 20))
        return str


    @staticmethod
    def get_source_block_calc_func(layerName):
        return CSourceBlockCalculator.get_source_block_calc_func(layerName)

    # @property
    # def baseModel(self):
    #     import MnistModel2
    #
    #     return MnistModel2.CMnistModel2()

    def _initMainNet(self):
        import PyTorch.DansuhModel as PyTorchModel

        self.net = PyTorchModel.AlexNet()   # highest_layer=None, doubleSizeLayerNames=self.doubleSizeLayerNames)
        # dataset = CMnistDataset()
        # self.net.init(self.imageDataset, 'QtLogs/ImageNet')
        # self.net.batchSize = max(8 * getCpuCoreCount(), 32)

        self.netsCache = dict()

    def setLearnRate(self, learnRate):
        if not self.net:
            self._initMainNet()
        self.curModelLearnRate = learnRate

    def setLayerTrainable(self, layerName, trainable, allowCombinedLayers=True):
        model = self._getNet()  # (layerName)
        for layer in model.base_model.layers:
            if CImageNetVisWrapper._isMatchedLayer(layer.name, layerName, allowCombinedLayers):
                layer.trainable = trainable
                print("Layer '%s' made %strainable" % (layer.name, '' if trainable else 'not '))
        assert self.curModelLearnRate
        self._compileModel(self.curModelLearnRate)

    def getRecommendedLearnRate(self):
        # return 0.1      # SGD
        return 0.001    # Adam


    def printProgress(self, str):
        with open('QtLogs/progress.log', 'a') as file:
            file.write(str + '\n')

    # Converting to channels first, as VisQtMain expects (batch, channels, y, x)
    def _transposeToOutBatchDims(self, activations):
        if len(activations.shape) == 5:    # Something like (batch, y, x, 4, channels / 4)
            if activations.shape[4] == 1:
                activations = np.squeeze(activations, 4)
            elif activations.shape[3] == 1:
                activations = np.squeeze(activations, 3)
            else:
                activations = activations.transpose((0, -1, 1, 2, 3))

        if len(activations.shape) == 4:
            activations = activations.transpose((0, -1, 1, 2))
        elif len(activations.shape) == 3:
            activations = activations.transpose((0, -1, 1))
        return activations

    # LayerNames - list of names or one name
    def doubleLayerWeights(self, layerNames, allowCombinedLayers=True):
        if not self.net:
            # No net and no problems
            self.doubleSizeLayerNames += layerNames
            return

        if not isinstance(layerNames, list):
            layerNames = [layerNames]
        for layerName in layerNames:
            if layerName in self.doubleSizeLayerNames:
                raise Exception("Layer '%s' already has doubled weights" % layerName)

        oldModel = self._getNet()
        self.doubleSizeLayerNames += layerNames
        self.net = None
        newModel = self._getNet()

        for layerInd, oldLayer in enumerate(oldModel.base_model.layers):
            allWeights = oldLayer.get_weights()
            # assert len(allWeights) == 1
            # weights = allWeights[0]
            # newAllWeights = []
            if not allWeights:
                continue

            for layerName in layerNames:
                # if isinstance(oldLayer.input, list):
                #     print('%s - %s' % (oldLayer.name, ','.join([l.name for l in oldLayer.input])))
                # else:
                #     print('%s - %s' % (oldLayer.name, oldLayer.input.name))

                newLayer = newModel.base_model.get_layer(index=layerInd)
                pos = -1
                while oldLayer.name.find('_', pos + 1) > pos:
                    pos = oldLayer.name.find('_', pos + 1)
                if pos < 0:
                    pos = len(oldLayer.name)
                assert newLayer.name[:pos] == oldLayer.name[:pos]
                allNewWeights = newLayer.get_weights()

                changed = False
                for i in range(len(allWeights)):
                    if allNewWeights[i].shape != allWeights[i].shape:
                        weights = allWeights[i]     # E.g. [5, 5, 48, 96]
                        stdDev = np.std(weights)
                        # mean = np.mean(weights)
                        print('%s: %s -> %s' % (oldLayer.name, allWeights[i].shape, allNewWeights[i].shape))
                        for axis in range(len(allWeights[i].shape)):
                            if allNewWeights[i].shape[axis] != allWeights[i].shape[axis]:
                                baseDelta = np.random.normal(loc=0, scale=stdDev / 128, size=weights.shape)

                                inds = np.arange(0, weights.shape[axis])
                                indsX2 = np.stack([inds, inds]).reshape(inds.shape[0] * 2, order='F')
                                weights = getSubarrByIndices(weights, axis, indsX2)
                                if not oldLayer.name.find('batch_norm') >= 0:
                                    weights /= 2

                                delta = np.zeros(weights.shape)
                                assignSubarrByIndices(delta, baseDelta, axis, inds * 2)
                                assignSubarrByIndices(delta, -baseDelta, axis, inds * 2 + 1)
                                weights += delta
                        allWeights[i] = weights
                        # changed = True
            newLayer.set_weights(allWeights)

        print('Doubling done')
        self.netsCache = dict()
        self.activationCache.clear()
        assert self.curModelLearnRate
        self._compileModel(self.curModelLearnRate)
        self.printProgress('Doubled layers: %s' % (', '.join(self.doubleSizeLayerNames)))


    def _getNet(self, highestLayer = None):
        if not self.net:
            self._initMainNet()

        if highestLayer is None:
            return self.net
        else:
            if not highestLayer in self.netsCache:
                import ImageNet

                self.netsCache[highestLayer] = ImageNet.CImageRecognitionNet(highestLayer, base_model=self.net)
            return self.netsCache[highestLayer]

    def _isMatchedLayer(layerName, layerToFindName, allowCombinedLayers):
        return layerName == layerToFindName or \
                (allowCombinedLayers and layerName[ : len(layerToFindName) + 1] == layerToFindName + '_')

    # def _compileModel(self, learnRate):
    #     from keras.optimizers import Adam, SGD
    #     from tensorflow_addons.optimizers import AdamW
    #     from keras.losses import categorical_crossentropy
    #
    #     # self.optimizer = SGD(lr=learnRate, decay=5e-4, momentum=0.9, nesterov=True)
    #     # self.optimizer = Adam(learning_rate=learnRate, decay=2.5e-4, epsilon=1e-6)
    #     self.optimizer = AdamW(learning_rate=learnRate, weight_decay=2.5e-4, epsilon=1e-6)
    #     # It's possible to turn off layers' weights updating with layer.trainable = False/
    #     # It requires model.compile for changes to take effect
    #     self.net.model.compile(optimizer=self.optimizer,
    #                            loss=categorical_crossentropy,  # loss='mse',
    #                            metrics=['accuracy'])
    #
