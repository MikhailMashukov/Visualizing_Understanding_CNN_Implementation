from __future__ import absolute_import, division, print_function, unicode_literals

import copy
# import datetime
import math
import random
import os
import psutil
# import time
import numpy as np
import warnings
import torch
import torch.nn.functional as F

import DeepOptions
import DataCache
from MyUtils import *
from MnistNetVisWrapper import CBaseLearningCallback, getSavedNetEpochs
import AlexNetVisWrapper
from ImageNetsVisWrappers import CImageNetPartDataset, CSourceBlockCalculator
import PyTorch.DansuhModel
import PyTorch.PyTImageModels
import PyTorch.PyTImageModel3
import PyTorch.PyTResNets
import PyTorch.PyTFractMPResNet

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class CPyTorchImageNetVisWrapper:
    def __init__(self, imageDataset=None, activationCache=None):
        self.name = 'image'
        self.weightsFileNameTempl = 'QtLogs/checkpoints/PyTImWeights_Epoch%d.pkl'
        self.optimStateFileNameTempl = 'QtLogs/checkpoints/PyTImOptState_Epoch%d.pkl'
        # self.weightsFileNameTempl = 'PyTLogs/checkpoints/PyTImWeights_Epoch%d.h5'
        self.imageCache = DataCache.CDataCache(256 * getCpuCoreCount())
        self.imageDataset = CImageNetPartDataset(self.imageCache) if imageDataset is None else imageDataset
        # self.imageDataset = CAugmentedMnistDataset(CImageNetPartDataset()) if imageDataset is None else imageDataset
        self.net = None
        self.batchSize = 64
        self.netPreprocessStageName = 'net'  # for self.net = alexnet.AlexNet()
        self.netImageSize = 224 if DeepOptions.modelClass.lower().find('resnet') >= 0 else 227
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

    def getImageActivations(self, layerName, imageNum, epochNum=None, allowCombinedLayers=True):
        return self.getImagesActivations_Batch(layerName, [imageNum], epochNum, allowCombinedLayers)

    def getImagesActivations_Batch(self, layerName, imageNums, epochNum=None, allowCombinedLayers=True,
                                   augment=False):
        if epochNum is None or epochNum < 0:
            epochNum = self.curEpochNum

        batchActs = [None] * len(imageNums)
        images = []
        for i in range(len(imageNums)):
            imageNum = imageNums[i]
            itemCacheName = 'act_%s_%d_%d%s' % (layerName, imageNum, epochNum, '_aug' if augment else '')
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

        model = self._getNet(layerName, allowCombinedLayers)
        if epochNum != self.curEpochNum:
            self.loadState(epochNum)
        imageData = np.stack(images, axis=0)
        imageData = np.transpose(imageData, (0, 3, 1, 2))        # Becomes (images, channels, x, y)
        # print("Predict data prepared")
#         print('imageData dtype', imageData.dtype, ', shape', torch.from_numpy(imageData).shape)
        if augment:
            imageData = self._prepareActivationsAugmentedImages(imageData)
#         print('imageData min', imageData.min(), ' max', imageData.max())

        model.eval()
        imageData = imageData + np.zeros(imageData.shape, dtype=np.float32)   # Workaround for "max_pool2d_with_indices_out_cuda_frame failed with error code 0"
        pytImageData = torch.from_numpy(imageData).cuda()
        pytImageData /= 128                        # Approximately (std is about 0.25)
        with torch.set_grad_enabled(False):
            activations = model.forward(pytImageData)   # np.expand_dims(imageData, 0), 3))
        # print("Predicted")
            activations = activations.cpu().numpy()
        if augment:
            activations = self._combineActivationsAugmentedImages(activations, len(imageNums))

#         if self.netPreprocessStageName == 'net':
#             activations = self._transposeToOutBatchDims(activations)
        # print('Activations', activations.shape)

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
#         print('Output prepared')
        return np.stack(batchActs, axis=0)

    # Takes (images, channels, x, y). Has big memory leak (260 GB / 200000 source images)
    def _prepareActivationsAugmentedImages(self, imageData):
        import scipy.ndimage
        import skimage.transform

        # imageSize = list(imageData.shape)[3:]
        augDataList = [imageData]
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            curData = np.pad(imageData, ((0, 0), (0, 0),
                        (-dx if dx < 0 else 0, dx if dx > 0 else 0),
                        (-dy if dy < 0 else 0, dy if dy > 0 else 0)))

            curData = curData[:, :, (dx if dx >= 0 else 0) : (dx if dx < 0 else curData.shape[2]),
                                    (dy if dy >= 0 else 0) : (dy if dy < 0 else curData.shape[3])]
            augDataList.append(np.array(curData))    # Copying here partially fixes memory leak

#         rotatedData = scipy.ndimage.rotate(imageData, 10, axes=(3, 2), reshape=False)
# #         print('rotatedData', rotatedData.shape)
#         mult = 20
#         for dx, dy in [(-mult, -mult), (mult, mult), (-mult, mult), (mult, -mult)]:
#             curData = rotatedData[:, :, (dx if dx >= 0 else 0) : (dx if dx < 0 else rotatedData.shape[2]),
#                                         (dy if dy >= 0 else 0) : (dy if dy < 0 else rotatedData.shape[3])]
# #             print('curData', curData.shape)
#             curData = skimage.transform.resize(curData, imageData.shape)
#             augDataList.append(curData)
#             if (dx, dy) == (mult, mult):
#                 rotatedData = scipy.ndimage.rotate(imageData, -10, axes=(3, 2), reshape=False)

        return np.concatenate(augDataList, axis=0)

    def _combineActivationsAugmentedImages(self, imageData, sourceImageCount):
        shape = list(imageData.shape)
        data = np.reshape(imageData, [sourceImageCount, shape[0] // sourceImageCount] + shape[1:],
                          order='F')
        data = np.mean(data, axis=1)
        return data


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

        weights = weights.cpu().numpy()      # E.g. [96, 3, 11, 11]
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

            # trainImageNums = np.arange(1, self.imageDataset.getImageCount('train') + 1)
            # testImageNums = np.arange(1, self.imageDataset.getImageCount('test') + 1)
            #     # TODO: actually doesn't pass to image dataset now

            # if 1:
                # if not options.trainImageNums is None:
                #     trainImageNums = options.trainImageNums
                #     if options.additTrainImageCount > 0:
                #         trainImageNums = np.concatenate([trainImageNums, np.random.randint(
                #                 low=1, high=self.imageDataset.getImageCount('train'),
                #                 size=options.additTrainImageCount)])

            for _ in range(int(math.ceil(iterCount / 100))):
                self.printProgress('Learn. rate %.3g, batch %d' % \
                        (options.learnRate, self.getBatchSize()))
                epochImageCount = 5000
                if 1:
                    if iterCount > 500:
                        if epochNum < 4:
                            pass
                        elif 2 << (epochNum - 4) <= 10:
                            epochImageCount = epochImageCount << (epochNum - 3)
                        else:
                            epochImageCount *= 10
                # if DeepOptions.imagesMainFolder.lower().find('imagenet') >= 0 and \
                #    DeepOptions.imagesMainFolder != 'ImageNetPart':
                #     epochImageCount *= 2

                if self.curModelLearnRate != options.learnRate:
                    self.setLearnRate(options.learnRate)
                    print('Learning rate switched to %f' % options.learnRate)

                # print('Train images: ', trainImageNums)
                # if epochNum == 0:
                #     infoStr = self.net.doLearning(1, callback,
                #             trainImageNums, testImageNums,
                #             epochImageCount, self.curEpochNum)
                # else:
                #     infoStr = self.net.doLearningWithPrevDataset(1, callback,
                #             epochImageCount, self.curEpochNum)
                infoStr = self._doLearning_Internal(epochImageCount)
                self.curEpochNum += 1
                epochNum += 1
                infoStr = 'Epoch %d: %s' % (self.curEpochNum, infoStr)
                if epochNum < 3: #  or epochNum % 10 == 0:
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

        # infoStr = "%s, last %d epochs: %.4f s" % \
        #           (infoStr, epochCount,
        #            (datetime.datetime.now() - groupStartTime).total_seconds())

        # self.activationCache.clear()
        return infoStr

    def _doLearning_Internal(self, epochImageCount):
        testImageCount = epochImageCount // 6

        batchCount = epochImageCount // self.batchSize
        lossSum = 0
        accuracySum = 0
        self.net.train()
        for batchNum in range(batchCount):
            imgs, classes_Cpu = self.getTrainBatch()
            imgs, classes = imgs.to(self.pytDevice), classes_Cpu.to(self.pytDevice)

            # calculate the loss
            output = self.net(imgs)
            loss = F.cross_entropy(output, classes)      # Averaged for entire batch

            # update the parameters
            self.pytOptimizer.zero_grad()
            loss.backward()
            self.pytOptimizer.step()

            lossSum += loss
            _, preds = torch.max(output.cpu(), 1)
            accuracySum += torch.sum(preds == classes_Cpu)

        print('%d batches processed' % batchCount)
        infoStr = "loss %.7g" % (float(lossSum) / batchCount)
        infoStr += ", acc %.5f" % (float(accuracySum) / self.batchSize / batchCount)
        self.lastEpochResult = {'TrainLoss': float(lossSum) / batchCount,
                                'TrainAcc': float(accuracySum) / self.batchSize / batchCount}

        self.net.eval()
        self.pytOptimizer.zero_grad()
        batchCount = testImageCount // self.batchSize
#         for batchNum in range(batchCount):
#                 imgs, classes_Cpu = self.getTestBatch()
#                 imgs, classes = imgs.to(self.pytDevice), classes_Cpu.to(self.pytDevice)
#                 with torch.set_grad_enabled(False):
#                     output = self.net(imgs)
        if 1:
            lossSum = 0
            accuracySum = 0
            for batchNum in range(batchCount):
                imgs, classes_Cpu = self.getTestBatch()
                imgs, classes = imgs.to(self.pytDevice), classes_Cpu.to(self.pytDevice)
                with torch.set_grad_enabled(False):
                    output = self.net(imgs)
                    loss = F.cross_entropy(output, classes)

                lossSum += loss
                _, preds = torch.max(output.cpu(), 1)
#                 print('classes', classes_Cpu)
#                 print('preds', preds)
#                 print(torch.sum(preds == classes_Cpu))
                accuracySum += torch.sum(preds == classes_Cpu)

            print('%d val. batches processed' % batchCount)
            infoStr += ", val. loss %.7f" % (float(lossSum) / batchCount)
            infoStr += ", val. acc %.7f" % (float(accuracySum) / self.batchSize / batchCount)
            self.lastEpochResult.update({'TestLoss': float(lossSum) / batchCount,
                                         'TestAcc': float(accuracySum) / self.batchSize / batchCount})
        return infoStr

    def getSavedNetEpochs(self):
        return getSavedNetEpochs(self.weightsFileNameTempl.replace('%d', '*'))

    def saveState(self, saveCache=True):
        if saveCache:
            self.saveCacheState()
        try:
            if not self.net is None:
                dir = os.path.split(self.weightsFileNameTempl)[0]
                if not os.path.exists(dir):
                    os.makedirs(dir)

                if self.curEpochNum % 20 != 0:
                    self.net.saveState(self.weightsFileNameTempl % self.curEpochNum)
                else:
                    self.net.saveState(self.weightsFileNameTempl % self.curEpochNum,
                            {'optimizer': self.pytOptimizer.state_dict()},       # Occupies too much space, 2 times more than weights
                            self.optimStateFileNameTempl % self.curEpochNum)
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
            additInfo = self.net.loadState(self.weightsFileNameTempl % epochNum)
            if not additInfo:
                additInfoFileName = self.optimStateFileNameTempl % epochNum
                if os.path.exists(additInfoFileName):
                    additInfo = self.net.loadStateAdditInfo(additInfoFileName)
            if 'optimizer' in additInfo:
                self.pytOptimizer.load_state_dict(additInfo['optimizer'])
                print('Optimizer state loaded')
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


    # @staticmethod
    def get_source_block_calc_func(self, layerName):
        # return CSourceBlockCalculator.get_source_block_calc_func(layerName)
        if DeepOptions.modelClass.lower().find('resnet') >= 0:
            # return PyTorch.PyTResNets.ResNet.CSourceBlockCalculator.get_source_block_calc_func(layerName)
            return self._getNet().get_source_block_calc_func(layerName)

        if layerName == 'conv_u_2':
            layerName = 'conv_2'
        elif layerName == 'conv_u_3':
            layerName = 'conv_3'
        if DeepOptions.modelClass == 'ImageModel4':
            return PyTorch.PyTImageModel3.ImageModel4_ConnectedTowers.CSourceBlockCalculator.get_source_block_calc_func(layerName)
        return AlexNetVisWrapper.CAlexNetVisWrapper.get_source_block_calc_func(layerName)

    # @property
    # def baseModel(self):
    #     import MnistModel2
    #
    #     return MnistModel2.CMnistModel2()

    def _initMainNet(self):
        import torch
        from PyTorch.lookahead import Lookahead

        self.pytDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if DeepOptions.modelClass == 'AlexnetModel':
            self.net = PyTorch.DansuhModel.AlexNet(num_classes=DeepOptions.classCount).to(self.pytDevice)
                # highest_layer=None, doubleSizeLayerNames=self.doubleSizeLayerNames)
        elif DeepOptions.modelClass == 'AlexnetModel4':
            self.net = PyTorch.PyTImageModels.AlexNet4(num_classes=DeepOptions.classCount).to(self.pytDevice)
        elif DeepOptions.modelClass == 'ImageModel3':
            self.net = PyTorch.PyTImageModel3.ImageModel3_Deeper(num_classes=DeepOptions.classCount).to(self.pytDevice)
        elif DeepOptions.modelClass == 'ImageModel4':
            self.net = PyTorch.PyTImageModel3.ImageModel4_ConnectedTowers(num_classes=DeepOptions.classCount).to(self.pytDevice)
        elif DeepOptions.modelClass == 'ResNet18':
            self.net = PyTorch.PyTResNets.resnet18(num_classes=DeepOptions.classCount).to(self.pytDevice)
        elif DeepOptions.modelClass == 'MyResNet':
            self.net = PyTorch.PyTResNets.my_resnet(num_classes=DeepOptions.classCount).to(self.pytDevice)
        elif DeepOptions.modelClass == 'FractMPResNet':
            self.net = PyTorch.PyTFractMPResNet.fract_max_pool_resnet(num_classes=DeepOptions.classCount).to(self.pytDevice)
        else:
            raise Exception('Unknown model class %s' % DeepOptions.modelClass)

        if 0:
            self.pytOptimizer = torch.optim.AdamW(params=self.net.parameters())
        else:
            print('SGD')
            self.pytOptimizer = torch.optim.SGD(params=self.net.parameters(), lr=self.getRecommendedLearnRate(),
                                                momentum=0.9, weight_decay=1e-4)
        self.pytBaseOptimizer = self.pytOptimizer
        self.pytOptimizer = Lookahead(self.pytOptimizer, k=5, alpha=0.5)

        self.netsCache = [dict(), dict()]
        self.setBatchSize(self.batchSize)

    def setBatchSize(self, batchSize):
        import PIL
        import torchvision.datasets as PyTorchDatasets
        import torchvision.transforms as transforms
        from torch.utils import data

        self.batchSize = batchSize
        datasetFolder = '%s/train' % (DeepOptions.imagesMainFolder)
        self.pytTrainDataset = PyTorchDatasets.ImageFolder(datasetFolder, transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.8, 1.3)),
                # transforms.RandomRotation(degrees=30, resample=PIL.Image.BICUBIC,
                #         expand=True, fill=(124, 117, 104)),
#                 transforms.CenterCrop(256),
                transforms.RandomCrop(self.netImageSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        self.pytTrainDataLoader = data.DataLoader(self.pytTrainDataset,
                shuffle=True,
                pin_memory=True,
                num_workers=16,
                drop_last=True,
                batch_size=self.batchSize)
        self.pytTrainDataIt = iter(self.pytTrainDataLoader)

        datasetFolder = '%s/test' % (DeepOptions.imagesMainFolder)
        self.pytTestDataset = PyTorchDatasets.ImageFolder(datasetFolder, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.netImageSize),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        self.pytTestDataLoader = data.DataLoader(self.pytTestDataset,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
                drop_last=True,
                batch_size=self.batchSize)
        self.pytTestDataIt = iter(self.pytTestDataLoader)
        print('Images loaders initialized. Batch size', self.batchSize)
#         print('First batch: ', next(self.pytTrainDataIt)[1])

    def getTrainBatch(self):
        try:
#             if random.random() < 5e-4:
#                 self.pytTrainDataIt = iter(self.pytTrainDataLoader)
#                 print('First batch: ', next(self.pytTrainDataIt)[1])
            return next(self.pytTrainDataIt)    # imgs, classes
        except StopIteration:
            self.pytTrainDataIt = iter(self.pytTrainDataLoader)
#             print('First batch: ', next(self.pytTrainDataIt)[1])
            return next(self.pytTrainDataIt)

    def getTestBatch(self):
        try:
            return next(self.pytTestDataIt)    # imgs, classes
        except StopIteration:
            self.pytTestDataIt = iter(self.pytTestDataLoader)
            return next(self.pytTestDataIt)


    def setLearnRate(self, learnRate):
        if not self.net:
            self._initMainNet()
        for g in self.pytOptimizer.param_groups:
            g['lr'] = learnRate
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

    def getBatchSize(self):
        return self.batchSize


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
        self.netsCache = [dict(), dict()]
        self.activationCache.clear()
        assert self.curModelLearnRate
        self._compileModel(self.curModelLearnRate)
        self.printProgress('Doubled layers: %s' % (', '.join(self.doubleSizeLayerNames)))


    def _getNet(self, highestLayer=None, allowCombinedLayers=False):
        if not self.net:
            self._initMainNet()

        if highestLayer is None:
            return self.net
        else:
            allowFlag = 1 if allowCombinedLayers else 0
            if not highestLayer in self.netsCache[allowFlag]:
                if DeepOptions.modelClass == 'AlexnetModel':
                    self.netsCache[allowFlag][highestLayer] = \
                            PyTorch.DansuhModel.CutAlexNet(self.net, highestLayer)
                elif DeepOptions.modelClass == 'AlexnetModel4':
                    self.netsCache[allowFlag][highestLayer] = \
                            PyTorch.PyTImageModels.AlexNet4.CutVersion(self.net, highestLayer, allowCombinedLayers)
                elif DeepOptions.modelClass == 'ImageModel3':
                    self.netsCache[allowFlag][highestLayer] = \
                            PyTorch.PyTImageModel3.ImageModel3_Deeper.CutVersion(self.net, highestLayer, allowCombinedLayers)
                elif DeepOptions.modelClass == 'ImageModel4':
                    self.netsCache[allowFlag][highestLayer] = copy.copy(self.net)
                    self.netsCache[allowFlag][highestLayer].highestLayerName = highestLayer
                elif DeepOptions.modelClass.lower().find('resnet') >= 0:
                    self.netsCache[allowFlag][highestLayer] = copy.copy(self.net)
                    self.netsCache[allowFlag][highestLayer].setHighestLayer(highestLayer)

            return self.netsCache[allowFlag][highestLayer]

    def _isMatchedLayer(layerName, layerToFindName, allowCombinedLayers):
        return layerName == layerToFindName or \
                (allowCombinedLayers and layerName[ : len(layerToFindName) + 1] == layerToFindName + '_')
