import copy
import datetime
import matplotlib
matplotlib.use('AGG')
matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvas
    # +FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
# from matplotlib import cm
# import pickle
import math
import numpy as np
# import PyQt4.Qt
# from PyQt4 import QtCore, QtGui
import os
# import random
# import sys
# import time

# sys.path.append(r"../Qt_TradeSim")
# import AlexNetVisWrapper
# import MnistNetVisWrapper
from MyUtils import *
from VisUtils import *

class TMultActOpsOptions:
    topCount = 16
    oneImageMaxTopCount = 6
    minDist = 3
    batchSize = 16 * getCpuCoreCount()
    embedImageNums = False

class CMultActTopsCalculator(TMultActOpsOptions):
    # c_channelMargin = 2
    # c_channelMargin_Top = 5
    # AllEpochs = -2

    def __init__(self, mainWindow, activationCache, netWrapper=None):
        self.mainWindow = mainWindow
        # # self.netWrapper = AlexNetVisWrapper.CAlexNetVisWrapper()
        # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper()
        self.netWrapper = netWrapper
        self.activationCache = activationCache
        self.imageDataset = self.netWrapper.getImageDataset()

    def onShowActTopsPressed(self):
        # import alexnet

        self.startAction(self.onShowActTopsPressed)
        imageNum = self.getSelectedImageNum()
        sourceImageData = self.imageDataset.getImage(imageNum, 'cropped')
        # alexNetImageData = self.imageDataset.getImage(imageNum, 'net')
        layerName = self.blockComboBox.currentText()
        activations = self.netWrapper.getImageActivations(layerName, imageNum)
        (chanCount, colCount) = todo
        activations = activations[0][:chanCount]
        if len(activations.shape) == 1:
            return
            # activations = np.expand_dims(activations, axis=0)

        sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(layerName)
        if sourceBlockCalcFunc is None:
            return
        # colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
        resultList = []
        # if layerName in ['conv_1', 'conv_2']
        #     activations[0, 22:25, 0] = 100     #d_
        for chanInd in range(activations.shape[0]):
            vals = attachCoordinates(activations[chanInd])   # Getting list of (x, y, value)
            sortedVals = vals[:, vals[2, :].argsort()]
            sourceBlock = sourceBlockCalcFunc(int(sortedVals[0, -1]), int(sortedVals[1, -1]))
            resultList.append(sourceImageData[sourceBlock[0] : sourceBlock[2], sourceBlock[1] : sourceBlock[3], :])
            # sourceUpperLeft = (int(sortedVals[0, -1]) * mult, int(sortedVals[1, -1]) * mult)
            # resultList.append(sourceImageData[sourceUpperLeft[1] : sourceUpperLeft[1] + size,
            #                                   sourceUpperLeft[0] : sourceUpperLeft[0] + size, :])

            # selectedList = []
            # i = vals.shape[0] - 1
            # while len(selectedList) < 9:
            #     curVal = vals[i]
            #     found = False
            #     for prevVal in selectedList
        # maxSize = 0
        # for result in resultList:
        #     if maxSize < max(result.shape[0:1]):
        #         maxSize = max(result.shape[0:1])
        # resultList2 = []
        # for result in resultList:
        #     if maxSize != result.shape[0] or maxSize != result.shape[1]:
        #         padded = np.pad(result, ((0, maxSize - result.shape[0]),
        #                                  (0, maxSize - result.shape[1]), (0, 0)),
        #                         constant_values=255)
        #     else:
        #         padded = result
        #     resultList2.append(padded)
        resultList = padImagesToMax(resultList)
        data = np.stack(resultList, axis=0)
        data = layoutLayersToOneImage(data, colCount, self.mainWindow.c_channelMargin)

        ax = self.figure.add_subplot(self.gridSpec[1, 0])
        ax.clear()
        self.showImage(ax, data)
        self.canvas.draw()

    # Makes first stage of data preparation. Returns (resulting array, number of processed images).
    # The returned array can be not finished in case of cancelling through callback
    def calcBestSourceCoords(self):
        bestSourceCoords = [[] for _ in range(self.chanCount)]
            # [layerNum][resultNum (the last - the best)] -> (imageNum, x at channel, y, value)
        lastActionStartTime = datetime.datetime.now()
        prevT = lastActionStartTime

        batchSize = 1
    #     batchSize = self.batchSize
        for batchNum in range((self.imageToProcessCount - 1) // batchSize + 1):
            imageNums = range(batchNum * batchSize + 1,
                              min((batchNum + 1) * batchSize, self.imageToProcessCount) + 1)
            if batchSize == 1:
                batchActivations = self.netWrapper.getImageActivations(
                        self.layerName, batchNum + 1, self.epochNum)
            else:
                print("Batch: ", ','.join(str(i) for i in imageNums))
                batchActivations = self.netWrapper.getImagesActivations_Batch(
                        self.layerName, imageNums, self.epochNum)
            if len(batchActivations.shape) == 2:
                batchActivations = np.expand_dims(np.expand_dims(batchActivations, axis=2), 2)

            for imageNum in imageNums:
                activations = batchActivations[imageNum - imageNums[0]][:self.chanCount]
                # self._saveBestActivationsCoords(bestSourceCoords, activations, self)

                # if layerNum <= 2:
                #     activations[0, 22:25, 0] = 100     #d_
                for chanInd in range(self.chanCount):
                    vals = attachCoordinates(activations[chanInd])    # Getting list of (x, y, value)
                    sortedVals = vals[:, vals[2, :].argsort()]
                    valsToSave = sortedVals[:, -self.oneImageMaxTopCount : ]    # Unfortunately without respect to min. distance
                    valsToSave[2, :] += np.mean(activations[chanInd]) / 3          # Adding influence of entire activation map
                    valsToSave = np.pad(valsToSave, ((1, 0), (0, 0)), constant_values=imageNum)
                    bestSourceCoords[chanInd].append(valsToSave)
                if imageNum % 16 == 0:
                    t = datetime.datetime.now()
                    if lastActionStartTime:
                        timeInfo = ', %.2f ms/image' % \
                            ((t - lastActionStartTime).total_seconds() * 1000 / imageNum)
                    else:
                        timeInfo = ''
                    timeInfo += ', last 16 - %.2f' % \
                            ((t - prevT).total_seconds() * 1000 / 16)
                    prevT = t
                    infoStr = 'Stage 1: image %d%s, %s in cache' % \
                            (imageNum, timeInfo, self.netWrapper.getCacheStatusInfo())
                        # progressCallback - like QMainWindow.TProgressIndicator
                    if not self.progressCallback.onMultActTopsProgress(infoStr, imageNum, bestSourceCoords):
                        return (bestSourceCoords, imageNum)

        return (bestSourceCoords, self.imageToProcessCount)
        # resultImage = self.showMultActTops(bestSourceCoords)
        # self.saveMultActTopsImage(resultImage)

    # def calcMultActTopsSourceCoords(self, self, progressCallback):
    #     bestSourceCoords = [[] for _ in range(self.chanCount)]
    #         # [layerNum][resultNum (the last - the best)] -> (imageNum, x at channel, y, value)
    #     prevT = datetime.datetime.now()
    #
    #     try:
    #         batchSize = 1
    #     #     batchSize = self.batchSize
    #         for batchNum in range((self.imageToProcessCount - 1) // batchSize + 1):
    #             imageNums = range(batchNum * batchSize + 1,
    #                               min((batchNum + 1) * batchSize, self.imageToProcessCount) + 1)
    #             if batchSize == 1:
    #                 batchActivations = self.netWrapper.getImageActivations(
    #                         layerName, batchNum + 1, self.epochNum)
    #             else:
    #                 print("Batch: ", ','.join(str(i) for i in imageNums))
    #                 batchActivations = self.netWrapper.getImagesActivations_Batch(
    #                         layerName, imageNums, self.epochNum)
    #             if len(batchActivations.shape) == 2:
    #                 batchActivations = np.expand_dims(np.expand_dims(batchActivations, axis=2), 2)
    #
    #             for imageNum in imageNums:
    #                 activations = self.getChannelsToAnalyze(batchActivations[imageNum - imageNums[0]])
    #                 # self._saveBestActivationsCoords(bestSourceCoords, activations, self)
    #
    #                 # if layerNum <= 2:
    #                 #     activations[0, 22:25, 0] = 100     #d_
    #                 for chanInd in range(self.chanCount):
    #                     vals = attachCoordinates(activations[chanInd])    # Getting list of (x, y, value)
    #                     sortedVals = vals[:, vals[2, :].argsort()]
    #                     valsToSave = sortedVals[:, -self.oneImageMaxTopCount : ]    # Unfortunately without respect to min. distance
    #                     valsToSave[2, :] += np.mean(activations[chanInd]) / 3          # Adding influence of entire activation map
    #                     valsToSave = np.pad(valsToSave, ((1, 0), (0, 0)), constant_values=imageNum)
    #                     bestSourceCoords[chanInd].append(valsToSave)
    #                 if imageNum % 16 == 0:
    #                     t = datetime.datetime.now()
    #                     if lastActionStartTime:
    #                         timeInfo = ', %.2f ms/image' % \
    #                             ((t - lastActionStartTime).total_seconds() * 1000 / imageNum)
    #                     else:
    #                         timeInfo = ''
    #                     timeInfo += ', last 16 - %.2f' % \
    #                             ((t - prevT).total_seconds() * 1000 / 16)
    #                     prevT = t
    #                     self.showProgress('Stage 1: image %d%s, %s in cache' % \
    #                                       (imageNum, timeInfo, \
    #                                        self.netWrapper.getCacheStatusInfo()))
    #                     if self.cancelling or self.exiting:
    #                         break
    #                     elif self.needShowCurMultActTops:
    #                         self.needShowCurMultActTops = False
    #                         resultImage = self.showMultActTops(bestSourceCoords, self.chanCount, options)
    #                         self.saveMultActTopsImage(resultImage, imageNum)
    #             if self.cancelling or self.exiting:
    #                 break

    def calcMultActTops_MultiThreaded(self):
        # My own implementation, from scratch, with images subblocks precision
        self.startAction(self.calcMultActTops_MultiThreaded)
        options = QtMainWindow.TMultActOpsOptions()
        # self.epochNum = self.getSelectedEpochNum()
        self.layerName = self.blockComboBox.currentText()
        self.embedImageNums = True
        self.imageToProcessCount = max(200 if self.netWrapper.name == 'mnist' else 20, \
                    self.getSelectedImageNum())
        self.epochNums = self.netWrapper.getSavedNetEpochs()
        self.threadCount = 4
        # threads = []
        calc_MultiThreaded(self.threadCount, self.calcMultActTopsThreadFunc,
                           mainWindow, self.epochNums, options)
        self.showProgress('%d threads finished' % self.threadCount)

    def calcMultActTopsThreadFunc(self, threadParams, options):
        downloadStats = threadParams.downloadStats
        self.showProgress('%s: started for %s%s' % \
              (threadParams.threadName, threadParams.ids[:20], \
               '...' if len(threadParams.ids) > 20 else ''))

        curThreadAdded = 0
        for epochNum in threadParams.ids:
            try:
                t0 = datetime.datetime.now()



            except Exception as errtxt:
                print('Exception at %s on epoch %d: %s' % (threadParams.threadName, epochNum, errtxt))

        with downloadStats.updateLock:
            downloadStats.finishedThreadCount += 1

        activations1 = self.netWrapper.getImageActivations(
                            layerName, 1, self.epochNum)
        activations1 = self.getChannelsToAnalyze(activations1[0])
        chanCount = activations1.shape[0]
        if self.checkMultActTopsInCache(chanCount, options):
            # No need to collect activations, everything will be taken from cache
            resultImage = self.showMultActTops(None)
            # self.saveMultActTopsImage(resultImage)
            return

        self.needShowCurMultActTops = False
        self.multActTopsButton.setText('Save current')
        try:
            self.multActTopsButton.clicked.disconnect()
        except e:
            pass
        self.multActTopsButton.clicked.connect(self.onShowCurMultActTopsPressed)

        # activations = self.getChannelsToAnalyze(self.netWrapper.getImageActivations(
        #           layerNum, 1, self.epochNum)[0])
        # print(activations)

        bestSourceCoords = None
            # [layerNum][resultNum (the last - the best)] -> (imageNum, x at channel, y, value)
        prevT = datetime.datetime.now()

        try:
            batchSize = 1
        #     batchSize = self.batchSize
            for batchNum in range((self.imageToProcessCount - 1) // batchSize + 1):
                imageNums = range(batchNum * batchSize + 1,
                                  min((batchNum + 1) * batchSize, self.imageToProcessCount) + 1)
                if batchSize == 1:
                    batchActivations = self.netWrapper.getImageActivations(
                            layerName, batchNum + 1, self.epochNum)
                else:
                    print("Batch: ", ','.join(str(i) for i in imageNums))
                    batchActivations = self.netWrapper.getImagesActivations_Batch(
                            layerName, imageNums, self.epochNum)
                if len(batchActivations.shape) == 2:
                    batchActivations = np.expand_dims(np.expand_dims(batchActivations, axis=2), 2)

                for imageNum in imageNums:
                    activations = self.getChannelsToAnalyze(batchActivations[imageNum - imageNums[0]])
                    if bestSourceCoords is None:
                        bestSourceCoords = [[] for _ in range(activations.shape[0])]
                    # self._saveBestActivationsCoords(bestSourceCoords, activations, options)

                    # if layerNum <= 2:
                    #     activations[0, 22:25, 0] = 100     #d_
                    for chanInd in range(activations.shape[0]):
                        vals = attachCoordinates(activations[chanInd])    # Getting list of (x, y, value)
                        sortedVals = vals[:, vals[2, :].argsort()]
                        valsToSave = sortedVals[:, -self.oneImageMaxTopCount : ]    # Unfortunately without respect to min. distance
                        valsToSave[2, :] += np.mean(activations[chanInd]) / 3          # Adding influence of entire activation map
                        valsToSave = np.pad(valsToSave, ((1, 0), (0, 0)), constant_values=imageNum)
                        bestSourceCoords[chanInd].append(valsToSave)
                    if imageNum % 16 == 0:
                        t = datetime.datetime.now()
                        if self.lastActionStartTime:
                            timeInfo = ', %.2f ms/image' % \
                                ((t - self.lastActionStartTime).total_seconds() * 1000 / imageNum)
                        else:
                            timeInfo = ''
                        timeInfo += ', last 16 - %.2f' % \
                                ((t - prevT).total_seconds() * 1000 / 16)
                        prevT = t
                        self.showProgress('Stage 1: image %d%s, %s in cache' % \
                                          (imageNum, timeInfo, \
                                           self.netWrapper.getCacheStatusInfo()))
                        if self.cancelling or self.exiting:
                            break
                        elif self.needShowCurMultActTops:
                            self.needShowCurMultActTops = False
                            resultImage = self.showMultActTops(bestSourceCoords)
                            self.saveMultActTopsImage(resultImage, imageNum)
                if self.cancelling or self.exiting:
                    break
        finally:
            self.multActTopsButton.setText(self.multActTopsButtonText)
            self.multActTopsButton.clicked.disconnect()
            self.multActTopsButton.clicked.connect(self.onShowMultActTopsPressed)

        if not self.exiting:
            resultImage = self.showMultActTops(bestSourceCoords, activations.shape[0], options)
            self.saveMultActTopsImage(resultImage)

    def checkMultActTopsInCache(self):
        for chanInd in range(self.chanCount):
            itemCacheName = 'MultAT_%s_%d_%d_%d' % \
                    (self.layerName, self.imageToProcessCount, self.epochNum, chanInd)
            cacheItem = self.activationCache.getObject(itemCacheName)
            if cacheItem is None:
                print("No %s in cache" % itemCacheName)
                return False
        return True

    # Takes prepared bestSourceCoords and/or data from cache and shows images
    # returns prepared image data (that it showed)
    def showMultActTops(self, bestSourceCoords, processedImageCount=None):
        if processedImageCount is None:
            processedImageCount = self.imageToProcessCount
        sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(self.layerName)
        # if not bestSourceCoords or len(bestSourceCoords[0]) == 0:
        #     return
        resultList = []
        topColCount = int(math.ceil(math.sqrt(self.topCount)))
        imageBorderValue = 0
        t0 = datetime.datetime.now()
        for chanInd in range(self.chanCount):
            itemCacheName = 'MultAT_%s_%d_%d_%d' % \
                    (self.layerName, processedImageCount, self.epochNum, chanInd)
            cacheItem = self.activationCache.getObject(itemCacheName)
            if not cacheItem is None:
                chanImageData = cacheItem
            else:
                chanImageData = self.buildChannelMultActTopImage(bestSourceCoords, chanInd,
                        sourceBlockCalcFunc, topColCount)
                self.activationCache.saveObject(itemCacheName, chanImageData)

                if (chanInd + 1) % 4 == 0:
                    t = datetime.datetime.now()
                    if (t - t0).total_seconds() >= 1:
                        infoStr = 'Stage 2: %d channels, %s in cache' % \
                                (chanInd + 1, self.netWrapper.getCacheStatusInfo())
                        if not self.progressCallback.onMultActTopsProgress(infoStr):
                            return None
                        t0 = t
            resultList.append(chanImageData)

        resultList = padImagesToMax(resultList, imageBorderValue)
        data = np.stack(resultList, axis=0)
        colCount = math.ceil(math.sqrt(self.chanCount) * 1.15 / 2) * 2
        chanBorderValue = 1 if data.dtype == np.float32 else 255
        data = layoutLayersToOneImage(data, colCount, self.mainWindow.c_channelMargin_Top, chanBorderValue)
        print("Top activations image built")

        # try:
        #     figure, axes = plt.subplots(223)
        #     figure.delaxes(axes)
        #     figure, axes = plt.subplots(224)
        #     figure.delaxes(axes)
        # except Exception as ex:
        #     print('Exception on subplot deletion: %s' % str(ex))
        ax = self.mainWindow.getMainSubplot()
        ax.clear()
        self.mainWindow.showImage(ax, data)
        self.mainWindow.canvas.draw()
        print("Canvas drawn")

        # import pickle

        # fileName = 'Data/BestActs/BestActs%d_%s_%dImages.dat' % \
        #         (self.topCount, self.layerName, int(maxImageNum))
        # with open(fileName, 'wb') as file:
        #     pickle.dump(bestSourceCoords, file)
        return data

    def buildChannelMultActTopImage(self, bestSourceCoords, chanInd,
                                    sourceBlockCalcFunc, topColCount):
        vals = np.concatenate(bestSourceCoords[chanInd], axis=1)       # E.g. 4 * 100
        if chanInd == 0:
            maxImageNum = np.max(vals[0, :])
        sortedVals = vals[:, vals[3, :].argsort()]

        selectedImageList = []
        selectedList = []                   # Will be e.g. 9 * 4
        i = sortedVals.shape[1] - 1
        while len(selectedList) < self.topCount and i >= 0:
            curVal = sortedVals[:, i]
            isOk = True
            for prevVal in selectedList:
                if curVal[0] == prevVal[0] and abs(curVal[1] - prevVal[1]) < self.minDist and \
                        abs(curVal[2] - prevVal[2]) < self.minDist:
                    isOk = False
                    break
            if isOk:
                sourceBlock = sourceBlockCalcFunc(int(curVal[1]), int(curVal[2]))
                curImageNum = int(curVal[0])
                imageData = self.imageDataset.getImage(curImageNum, 'cropped')
                blockData = imageData[sourceBlock[1] : sourceBlock[3], sourceBlock[0] : sourceBlock[2]]
                if self.embedImageNums and curImageNum <= 255 and imageData.max() > 1.01:
                    blockData[-1][-1] = curImageNum
                selectedImageList.append(blockData)
                selectedList.append(curVal)
            i -= 1
        imageBorderValue = 0  # 1 if selectedImageList[0].dtype == np.float32 else 255
        selectedImageList = padImagesToMax(selectedImageList, imageBorderValue)
        chanData = np.stack(selectedImageList, axis=0)
        chanImageData = layoutLayersToOneImage(chanData, topColCount, 1, imageBorderValue)
        bestSourceCoords[chanInd] = [np.stack(selectedList).transpose()]
        return chanImageData

    def saveMultActTopsImage(self, imageData, processedImageCount=None):
        from scipy.misc import imsave

        if processedImageCount is None:
            processedImageCount = self.imageToProcessCount
        if len(imageData.shape) == 3 and imageData.shape[2] == 1:
            imageData = np.squeeze(imageData, 2)
        dirName = 'Data/%s_%dChan_%dImages' % \
                 (self.layerName, self.chanCount, processedImageCount)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        # fileName = 'Data/top%d_%s_epoch%d_%dChan_%dImages.png' % \
        fileName = '%s/Top%d_Epoch%d.png' % \
                 (dirName, self.topCount, self.epochNum)
        imsave(fileName, imageData, format='png')

        # self.figure.savefig('Results/top%d_%s_%dChannels_%dImages.png' %
        #                         (self.topCount, layerName, activations.shape[0], imageCount),
        #                     format='png', dpi=resultImageShape[0] / 3)

    def showActTops_FromCsv(self, options):
        actMatrix = self.netWrapper.getImagesActivationMatrix(self.layerNum)
        chanCount = actMatrix.shape[1]
        if self.maxAnalyzeChanCount and chanCount > self.maxAnalyzeChanCount:
            chanCount = self.maxAnalyzeChanCount
        sortedValsList = []
        for chanInd in range(chanCount):
            sortedImageNums = np.flip(actMatrix[:, chanInd].argsort())[:self.topCount] + 1
            sortedValsList.append(sortedImageNums)
        imageNums = np.stack(sortedValsList, axis=0)
        self.showTopImages(imageNums, options)

    def showTopImages(self, imageNums, options):
        topColCount = int(math.ceil(math.sqrt(self.topCount)))
        chanCount = imageNums.shape[0]
        resultList = []
        t0 = datetime.datetime.now()
        for chanInd in range(chanCount):
            selectedImageList = []
            for imageNum in imageNums[chanInd][:self.topCount]:
                selectedImageList.append(self.imageDataset.getImage(imageNum, 'cropped'))
            chanData = np.stack(selectedImageList, axis=0)
            chanImageData = layoutLayersToOneImage(chanData, topColCount, 1)
            resultList.append(chanImageData)
            if (chanInd + 1) % 4 == 0:
                t = datetime.datetime.now()
                if (t - t0).total_seconds() >= 1:
                    self.showProgress('%d channels, %s in cache' % \
                                      (chanInd + 1, \
                                       self.netWrapper.getCacheStatusInfo()))
                    t0 = t

        data = np.stack(resultList, axis=0)
        colCount = math.ceil(math.sqrt(chanCount) * 1.15 / 2) * 2
        data = layoutLayersToOneImage(data, colCount, self.mainWindow.c_channelMargin_Top)

        ax = self.getMainSubplot()
        ax.clear()
        self.showImage(ax, data)
        self.canvas.draw()
        return data.shape[0:2]


