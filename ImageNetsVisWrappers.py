from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import math
import random
import os
# import time
import numpy as np

import DataCache
from MyUtils import *
from MnistNetVisWrapper import *
import AlexNetVisWrapper

# My own model for images recognition (classification)
class CImageNetVisWrapper:
    def __init__(self, imageDataset=None, activationCache=None):
        self.name = 'image'
        self.weightsFileNameTempl = 'QtLogs/ImagesWeights_Epoch%d.h5'
        # self.gradientsFileNameTempl = 'QtLogs/ImagesGrads_Epoch%d.dat'
        self.imageCache = DataCache.CDataCache(256 * getCpuCoreCount())
        self.imageDataset = CImageNetPartDataset(self.imageCache) if imageDataset is None else imageDataset
        # self.imageDataset = CAugmentedMnistDataset(CImageNetPartDataset()) if imageDataset is None else imageDataset
        self.net = None
        self.netPreprocessStageName = 'net'  # for self.net = alexnet.AlexNet()
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
        return ['conv_1', 'conv_2', 'conv_3', 'dense_1', 'dense_2']

    def getComponentNetLayers(self):
        return self.getNetLayersToVisualize()

    def getTowerCount(self):
        return 1

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
        activations = model.model.predict(imageData)   # np.expand_dims(imageData, 0), 3))

        if self.netPreprocessStageName == 'net':
            # Converting to channels first, as VisQtMain expects (batch, channels, y, x)
            if len(activations.shape) == 4:
                activations = activations.transpose((0, -1, 1, 2))
            elif len(activations.shape) == 3:
                activations = activations.transpose((0, -1, 1))
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
                self.activationCache.saveObject(itemCacheName, np.expand_dims(batchActs[i], 0))
                predictedI += 1
        assert predictedI == activations.shape[0]
        if len(images) == len(imageNums):
            return activations
        print('Output prepared')
        return np.stack(batchActs, axis=0)

    # Returns convolutions' multiplication weights (not bias) or similar for other layers
    def getMultWeights(self, layerName, allowCombinedLayers=False):  # , epochNum):
        model = self._getNet()

        try:
            layer = model.base_model.get_layer(layerName)
            allWeights = layer.get_weights()
        except:
            allWeights = None

        if allWeights:
            assert len(allWeights) == 1 or len(allWeights[0].shape) > len(allWeights[1].shape)
            weights = allWeights[0]
        else:
            if allowCombinedLayers:
                allWeights = []
                for layer in model.base_model.layers:
                    if layer.name.find(layerName + '_') >= 0:
                        allLayerWeights = layer.get_weights()
                        assert len(allLayerWeights) == 1 or len(allLayerWeights[0].shape) > len(allLayerWeights[1].shape)
                        allWeights.append(allLayerWeights[0])
                if not allWeights:
                    raise Exception('No weights found for combined layer %s' % layerName)
                allWeights = np.concatenate(allWeights, axis=3)
            else:
                raise Exception('No weights found for layer %s' % layerName)

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
            self.gradientTensors = getGradientTensors(self.net.model)
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

                if self.curModelLearnRate != options.learnRate:
                    self.setLearnRate(options.learnRate)
                    print('Learning rate switched to %f' % options.learnRate)

                trainImageNums = np.arange(1, self.imageDataset.getImageCount('train') + 1)
                testImageNums = np.arange(1, self.imageDataset.getImageCount('test') + 1)

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

                # print('Train images: ', trainImageNums)
                infoStr = self.net.doLearning(1, callback,
                                              trainImageNums, testImageNums,
                                              epochImageCount, self.curEpochNum)
                print(self.getCacheStatusInfo(True))
                self.curEpochNum += 1
                epochNum += 1
                infoStr = 'Epoch %d: %s' % (self.curEpochNum, infoStr)
                self.saveState(self.curEpochNum % 8 == 0)
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

    def saveState(self, saveCache=True):
        try:
            if saveCache:
                with open('Data/ImageNetVisActCache.dat', 'wb') as file:
                    self.activationCache.saveState_OpenedFile(file)
            if not self.net is None:
                self.net.model.save_weights(self.weightsFileNameTempl % self.curEpochNum)
        except Exception as ex:
            print("Error in saveState: %s" % str(ex))

    # def saveCurGradients(self):
    #     import pickle
    #
    #     with open(self.gradientsFileNameTempl % self.curEpochNum, 'wb') as file:
    #         gradientDict = self.getGradients()
    #         for name, value in gradientDict.items():
    #             pickle.dump(name, file)
    #             pickle.dump(value, file)

    # When it is desirable to initialize quickly (without tensorflow)
    def loadCacheState(self):
        try:
            with open('Data/ImageNetVisActCache.dat', 'rb') as file:
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

            if hasattr(self.net.model, 'weightsModel'):
                try:
                    self.net.model.weightsModel.load_weights(self.weightsFileNameTempl % epochNum)
                    print("Loaded weights for other model")
                except Exception as ex:
                    print("Error on loading weights to other model: %s" % str(ex))
                    self.net.model.load_weights(self.weightsFileNameTempl % epochNum)
            else:
                self.net.model.load_weights(self.weightsFileNameTempl % epochNum)
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
        if layerName == 'conv_1':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_conv_1_source_block
        elif layerName == 'conv_2':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_conv_2_source_block
        elif layerName == 'conv_3':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_conv_3_source_block
        elif layerName == 'conv_4':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_conv_4_source_block
        elif layerName == 'conv_5':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_conv_5_source_block
        elif layerName[:6] == 'dense_':
            return AlexNetVisWrapper.CAlexNetVisWrapper.get_entire_image_block
        else:
            return None

    # @property
    # def baseModel(self):
    #     import MnistModel2
    #
    #     return MnistModel2.CMnistModel2()

    def _initMainNet(self):
        # import alexnet

        # self.net = alexnet.AlexNet()
        # self.netDataFormat = 'channels_first'

        import ImageNet

        self.net = ImageNet.CImageRecognitionNet(None)
        # dataset = CMnistDataset()
        self.net.init(self.imageDataset, 'QtLogs/ImageNet')
        # # self.setLearnRate(0.1)
        # # if os.path.exists(self.weightsFileNameTempl):
        # #     self.net.model.load_weights(self.weightsFileNameTempl)
        # # self.net.model._make_predict_function()
        self.net.batchSize = max(8 * getCpuCoreCount(), 64)

        self.netsCache = dict()

    def setLearnRate(self, learnRate):
        from keras.optimizers import Adam, SGD

        # optimizer = SGD(lr=learnRate, decay=5e-6, momentum=0.9, nesterov=True)
        optimizer = Adam(learning_rate=learnRate, decay=5e-5)
        # It's possible to turn off layers' weights updating with layer.trainable = False/
        # It requires model.compile for changes to take effect
        self.net.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.curModelLearnRate = learnRate

    def getRecommendedLearnRate(self):
        # return 0.1      # SGD
        return 0.001    # Adam

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


class CImageNetPartDataset:
    def __init__(self, cache):
        self.trainSubset = CImageNetSubset('train', cache, self)
        self.testSubset = CImageNetSubset('test', cache, self)
        self.subsets = { 'train': self.trainSubset, 'test': self.testSubset }

    def getSubset(self, subsetName):
        return self.subsets[subsetName]

    def getImageFolder(self, subsetName='train'):
        return self.subsets[subsetName].mainFolder

    def getImage(self, imageNum, preprocessStage='net', subsetName='train'):
        return self.subsets[subsetName].getImage(imageNum, preprocessStage)

    def getImageCount(self, subsetName):
        return self.subsets[subsetName].getImageCount()

    def getImageLabel(self, imageNum, subsetName='train'):
        return self.subsets[subsetName].getImageLabel(imageNum)

    def getClassCount(self):
        return self.trainSubset.getClassCount()

    def getClassNameLabel(self, label):
        return self.trainSubset.getClassNameLabel(label)

    def getTfDataset(self, subsetName='train'):
        return self.subsets[subsetName].getTfDataset()

    def getNetSource(self, subsetName='train'): # TODO: to remove
        return self.subsets[subsetName].getNetSource()


    def checkAllSubsetsMatch(self):
        folders = None
        for subsetName, subset in self.subsets.items():
            if not subset.isLoaded():
                subset._loadData()

            if folders is None:
                folders = subset.folders
            else:
                if list(folders) != list(subset.folders):
                    print(list(folders), '\n%15s' % '', list(subset.folders))
                    raise Exception('%s images subset folders mismatch' % subsetName)


class CImageNetSubset:
    def __init__(self, subsetName, cache, parent):
        self.subsetName = subsetName
        self.mainFolder = 'ImageNetPart/%s' % self.subsetName
        self.foldersInfoCacheFileName = 'Data/ImageNet%sCache.dat' % self.subsetName
        self.cache = cache
        self.cachePackedImages = True       # Cache images in int8
        self.parent = parent
        self.folders = None          # List of folder names     # dict(folder) -> list of file names
        # self.imagesFolders = None    # ImageNum -> folder name
        self.imageNumLabels = None   # ImageNum -> label [0, category count)
        self.imagesFileNames = None  # ImageNum -> file path

        # try:
        #     import pandas
        #
        #     #with open('Data/ILSVRC2012_classes.txt', mode='r'):
        #     self.testLabels = pandas.read_table('Data/ILSVRC2012_classes.txt',
        #             dtype=int, header=None, squeeze=True)
        # except Exception as ex:
        #     print("Error on loading 'ILSVRC2012_classes.txt': %s" % str(ex))
        #     self.testLabels = np.ones([50000])

    # def getImageFilePath(self, imageNum):
    #     return 'ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG' % imageNum

    readImageCount =0   #d_
    def getImage(self, imageNum, preprocessStage='net'):
        self.readImageCount += 1
        # print('getImage %s %d' % (subsetName, self.readImageCount))
        itemCacheName = self._getImageCacheName(imageNum, preprocessStage)
        cacheItem = self.cache.getObject(itemCacheName)
        if not cacheItem is None:
            # if self.cachePackedImages:
            #     # print('cached', cacheItem.shape, cacheItem.dtype)
            #     return np.array(cacheItem, dtype=np.float32)
            return cacheItem

        # if not subsetName in ['train', 'test']:
        #     raise Exception("Invalid image subset '%s'" % subsetName)

        import alexnet_utils

        if self.folders is None:
            self._loadData()

        imgFileName = os.path.join(self.mainFolder, self.imagesFileNames[imageNum])
        img_size=(256, 256)
        crop_size=(227, 227)

        if preprocessStage == 'source':
            imageData = alexnet_utils.imread(imgFileName, pilmode='RGB')
        elif  preprocessStage == 'resized256':   # Resized, for alexnet, in uint8
            imageData = alexnet_utils.imread(imgFileName, pilmode='RGB')
            imageData = alexnet_utils.imresize(imageData, img_size)
            # imageData[:, :, 0] -= 123
            # imageData[:, :, 1] -= 116
            # imageData[:, :, 2] -= 103
        elif  preprocessStage == 'cropped':   # Cropped and resized, as for alexnet
                # but in uint8, without normalization and transposing back and forth.
                # Float32 lead to incorrect colors in imshow
            img_size=(256, 256)
            imageData = alexnet_utils.imread(imgFileName, pilmode='RGB')   # For imageio.imread
            imageData = alexnet_utils.imresize(imageData, img_size)
            imageData = imageData[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2,
                (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2, :]
            # imageData[:, [1, 4, 7], :] = [[255, 0, 0], [0, 200 ,0], [0, 0, 145]]
        else:
            if self.cachePackedImages:
                imageData = self.getImage(imageNum, 'cropped')
                imageData = np.array(imageData, dtype=np.float32)
                imageData[:, :, 0] -= 123
                imageData[:, :, 1] -= 116
                imageData[:, :, 2] -= 103
                # imageData[:, :, 0] -= 123.68
                # imageData[:, :, 1] -= 116.779
                # imageData[:, :, 2] -= 103.939
                if preprocessStage != 'net':
                    imageData = imageData.transpose((2, 0, 1))
                return imageData
            else:
                imageData = alexnet_utils.preprocess_image_batch([imgFileName])[0]

                if preprocessStage == 'net':
                    imageData = imageData.transpose((1, 2, 0))
                # else preprocessStage == 'net_channels_first'

        # if self.cachePackedImages:
        #     a = np.array(imageData, dtype=np.uint8)
        #     print('Image min %.3f, max %.3f,    int min %d, max %d' % \
        #           (imageData.min(), imageData.max(), a.min(), a.max()))
        #     self.cache.saveObject(itemCacheName, a)
        # else:
        #     self.cache.saveObject(itemCacheName, imageData)

        self.cache.saveObject(itemCacheName, imageData)
        return imageData

    def isLoaded(self):
        return not self.folders is None

    def getImageCount(self):
        if not self.isLoaded():
            self._loadData()
        return len(self.imageNumLabels) - 1

    # Labels here are class indices (0-based), imageNum is 1-based
    def getImageLabel(self, imageNum):
        if not self.isLoaded():
            self._loadData()
        return self.imageNumLabels[imageNum]

    def getClassCount(self):
        if not self.isLoaded():
            self._loadData()
        return len(self.folders)

    def getClassNameLabel(self, label):
        return self.folders[label]

    def getNetSource(self): # TODO: to remove
        return (self.imagesFileNames, self.imageNumLabels)

        # if self.test is None:
        #     self.loadData()
        # subset = self.train if type == 'train' else self.test
        if subsetName != 'test':
            print('Warning: no AlexNet train images')
        data = []
        for imageInd in range(1000):
            data.append(self.getImage(imageInd + 1, 'net'))
        return (np.stack(data, axis=0), self.testLabels)

    def getTfDataset(self):
        import tensorflow as tf

        # def _loadImage(imageNum):
        #     imageData = self.getImage(imageNum, 'net')
        #     return imageData
        #
        # def _tfLoadImage(imageNum):
        #     image = tf.py_function(_loadImage, [imageNum], tf.float32)
        #     return image

        imageNums = np.arange(1, self.getImageCount() + 1)
        # path_ds = tf.data.Dataset.from_tensor_slices(self.imagesFileNames)
        numDs = tf.data.Dataset.from_tensor_slices(imageNums)
        # image_ds = numDs.map(_tfLoadTrainImage, num_parallel_calls=1)
        # for n, image in enumerate(load_image_ds.take(7)):
        #     print(image)
        # if self.subsetName == 'test':
        #     print(imageNums)

        label_ds = tf.data.Dataset.from_tensor_slices(self.imageNumLabels[1:])
        ds = tf.data.Dataset.zip((numDs, label_ds))
        ds = ds.repeat()
        return ds

    def _loadData(self):
        import pickle

        try:
            with open(self.foldersInfoCacheFileName, 'rb') as file:
                self.folders = pickle.load(file)
                self.imageNumLabels = pickle.load(file)
                self.imagesFileNames = pickle.load(file)
        except Exception as ex:
            print("Error in %s CImageNetPartDataset._loadData: %s" % (self.subsetName, str(ex)))
            self._loadFilesTree()
            with open(self.foldersInfoCacheFileName, 'wb') as file:
                pickle.dump(self.folders, file)
                pickle.dump(self.imageNumLabels, file)
                pickle.dump(self.imagesFileNames, file)

    def _loadFilesTree(self):
        import glob
        # import re

        seed = 1
        self.folders = []
        self.imageNumLabels = []
        self.imagesFileNames = []
        sortedFolders = [folderName for folderName in os.listdir(self.mainFolder)]
        sortedFolders.sort()
        for folderName in sortedFolders:
            folderPath = os.path.join(self.mainFolder, folderName)
            if os.path.isdir(folderPath):
                print('Scanning images folder %s' % folderName)
                curNumLabel = len(self.folders)
                self.folders.append(folderName)

                for fileName in os.listdir(folderPath):
                    filePath = os.path.join(folderPath, fileName)
                    if os.path.isfile(filePath):
                        self.imageNumLabels.append(curNumLabel)
                        self.imagesFileNames.append('%s/%s' % (folderName, fileName))
        self.folders = np.array(self.folders)
        self.imageNumLabels = np.array(self.imageNumLabels)
        self.imagesFileNames = np.array(self.imagesFileNames)
        
        randomizer = random.Random(seed)
        inds = np.arange(len(self.imageNumLabels))
        randomizer.shuffle(inds)
        inds = np.concatenate(([0], inds))
        # self.folders = self.folders[inds]
        self.imageNumLabels = self.imageNumLabels[inds]
        self.imagesFileNames = self.imagesFileNames[inds]

        # for fileName in glob.glob(fileMask):
        #     fileName = fileName.lower()

        if self.subsetName == 'train':
            self.parent.checkAllSubsetsMatch()
        # else:
        #     print(self.imagesFileNames[:5], self.imageNumLabels[:5])

    def _getImageCacheName(self, imageNum, preprocessStage):
        return 'im_%d_%c_%s' %\
               (imageNum, self.subsetName[1], preprocessStage)
        # return 'im_%d_%c_%s_%s' %\
        #        (imageNum, subsetName[1], preprocessStage, 'i8' if self.cachePackedImages else 'f32')
