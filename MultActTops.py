import concurrent.futures
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
import queue
import _thread

# sys.path.append(r"../Qt_TradeSim")
# import AlexNetVisWrapper
# import MnistNetVisWrapper
from MyUtils import *
from VisUtils import *

class TMultActOpsOptions:
    topCount = 25
    oneImageMaxTopCount = 4
    minDist = 3
    batchSize = 16 * getCpuCoreCount()
    embedImageNums = False
    c_channelMargin = 2
    c_channelMargin_Top = 5

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

    # Makes first stage of data preparation. Returns (resulting array, number of processed images).
    # The returned array can be not finished in case of cancelling through callback
    def calcBestSourceCoords(self):
        bestSourceCoords = [[] for _ in range(self.chanCount)]
            # [layerNum][resultNum (the last - the best)] -> (imageNum, x at channel, y, value)
        lastActionStartTime = datetime.datetime.now()
        prevT = lastActionStartTime

        # batchSize = 1
        batchSize = self.batchSize
        for batchNum in range((self.imageToProcessCount - 1) // batchSize + 1):
            imageNums = range(batchNum * batchSize + 1,
                              min((batchNum + 1) * batchSize, self.imageToProcessCount) + 1)
            if batchSize == 1:
                try:
                    batchActivations = self.netWrapper.getImageActivations(
                            self.layerName, batchNum + 1, self.epochNum)
                except Exception as ex:
                    print("Error on batchActivations: %s" % (str(ex)))
            else:
                # print("Batch: ", ','.join(str(i) for i in imageNums))
                batchActivations = self.netWrapper.getImagesActivations_Batch(
                        self.layerName, imageNums, self.epochNum)
            if len(batchActivations.shape) == 2:
                batchActivations = np.expand_dims(np.expand_dims(batchActivations, axis=2), 2)

            for imageNum in imageNums:
                activations = batchActivations[imageNum - imageNums[0]][:self.chanCount]
                # self._saveBestActivationsCoords(bestSourceCoords, activations, self)

                # if layerNum <= 2:
                #     activations[0, 22:25, 0] = 100     #d_
                # print('1')
                means = np.mean(activations, axis=(1, 2)) / 3
                for chanInd in range(self.chanCount):
                    vals = attachCoordinates(activations[chanInd])    # Getting list of (x, y, value)
                    sortedVals = vals[:, vals[2, :].argsort()]
                    valsToSave = sortedVals[:, -self.oneImageMaxTopCount : ]    # Unfortunately without respect to min. distance
                    valsToSave[2, :] += means[chanInd]                # Adding influence of entire activation map
                    valsToSave = np.pad(valsToSave, ((1, 0), (0, 0)), constant_values=imageNum)
                    bestSourceCoords[chanInd].append(valsToSave)
                    # print('-ch-')
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

    # def calcMultActTops_MultiThreaded(self):
    #     # My own implementation, from scratch, with images subblocks precision
    #     self.startAction(self.calcMultActTops_MultiThreaded)
    #     options = QtMainWindow.TMultActOpsOptions()
    #     # self.epochNum = self.getSelectedEpochNum()
    #     self.layerName = self.blockComboBox.currentText()
    #     self.embedImageNums = True
    #     self.imageToProcessCount = max(200 if self.netWrapper.name == 'mnist' else 20, \
    #                 self.getSelectedImageNum())
    #     self.epochNums = self.netWrapper.getSavedNetEpochs()
    #     self.threadCount = 4
    #     # threads = []
    #     calc_MultiThreaded(self.threadCount, self.calcMultActTopsThreadFunc,
    #                        mainWindow, self.epochNums, options)
    #     self.showProgress('%d threads finished' % self.threadCount)


    def checkMultActTopsInCache(self):
        for chanInd in range(self.chanCount):
            itemCacheName = 'MultAT_%s_%d_%d_%d' % \
                    (self.layerName, self.imageToProcessCount, self.epochNum, chanInd)
            cacheItem = self.activationCache.getObject(itemCacheName)
            if cacheItem is None:
                print("No %s in cache" % itemCacheName)
                return False
        return True

    def buildMultActTopsImage(self, bestSourceCoords, processedImageCount=None):
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
        chanBorderValue = 1 if data.dtype == np.float32 else 255
        data = layoutLayersToOneImage(data, self.colCount, self.c_channelMargin_Top, chanBorderValue)
        print("Top activations image built")
        return data

    # Takes prepared bestSourceCoords and/or data from cache and shows images
    # returns prepared image data (that it showed)
    def showMultActTops(self, bestSourceCoords, processedImageCount=None):
        imageData = self.buildMultActTopsImage(bestSourceCoords, processedImageCount)

        # try:
        #     figure, axes = plt.subplots(223)
        #     figure.delaxes(axes)
        #     figure, axes = plt.subplots(224)
        #     figure.delaxes(axes)
        # except Exception as ex:
        #     print('Exception on subplot deletion: %s' % str(ex))
        ax = self.mainWindow.getMainSubplot()
        ax.clear()
        self.mainWindow.showImage(ax, imageData)
        self.mainWindow.canvas.draw()
        print("Canvas drawn")

        # import pickle

        # fileName = 'Data/BestActs/BestActs%d_%s_%dImages.dat' % \
        #         (self.topCount, self.layerName, int(maxImageNum))
        # with open(fileName, 'wb') as file:
        #     pickle.dump(bestSourceCoords, file)
        return imageData

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
                if blockData.shape[0] == 0 or blockData.shape[1] == 0:
                    # Reproduced with incorrect sorting in calcBestSourceCoords
                    print("0 block: %s, %d, %s" % (str(curVal), curImageNum, str(sourceBlock)))
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
        # colCount = math.ceil(math.sqrt(chanCount) * 1.15 / 2) * 2
        data = layoutLayersToOneImage(data, self.colCount, self.c_channelMargin_Top)

        ax = self.getMainSubplot()
        ax.clear()
        self.showImage(ax, data)
        self.canvas.draw()
        return data.shape[0:2]



class CMultiThreadedCalculator:
    def run(self, calculator, epochNums):
        self.mainCalculator = calculator
        self.activationsQueue = queue.Queue(maxsize=calculator.threadCount * 5)
        self.stopEvent = threading.Event()
        # self.mainWindowLock = _thread.allocate_lock()
        self.threadCrashed = False

        options = self.mainCalculator
        assert options.threadCount > 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=options.threadCount - 1) as executor:
            # executor.submit(producer, pipeline, event)
            for i in range(options.threadCount - 1):
                executor.submit(CMultiThreadedCalculator.workerThreadFunc, self, 'Thread %d' % i)

            batchSize = calculator.batchSize
            for epochNum in epochNums:
                # for batchNum in range((calculator.imageToProcessCount - 1) // batchSize + 1):
                #     imageNums = range(batchNum * batchSize + 1,
                #                       min((batchNum + 1) * batchSize, calculator.imageToProcessCount) + 1)
                #     # if batchSize == 1:
                #     #     batchActivations = calculator.netWrapper.getImageActivations(
                #     #             calculator.layerName, batchNum + 1, calculator.epochNum)
                #     # else:
                #     #     print("Batch: ", ','.join(str(i) for i in imageNums))
                #     batchActivations = calculator.netWrapper.getImagesActivations_Batch(
                #                 calculator.layerName, imageNums, calculator.epochNum)
                epochActivations = calculator.netWrapper.getImagesActivations_Batch(
                                calculator.layerName, range(1, calculator.imageToProcessCount + 1),
                                epochNum)
                print("Epoch %d activations prepared" % epochNum)
                if self.threadCrashed:
                    print("Crashing detected")
                    break
                self.activationsQueue.put((epochNum, epochActivations))
            self.stopEvent.set()

        print("Finished")

    # TODO: cancelling support
    def run_MultiProcess(self, calculator, epochNums):
        import multiprocessing

        self.mainCalculator = calculator
        self.stopEvent = threading.Event()
        # self.mainWindowLock = _thread.allocate_lock()
        # self.threadCrashed = False

        options = self.mainCalculator
        cleanedCalculator = copy.copy(self.mainCalculator)
        cleanedCalculator.mainWindow = None
        cleanedCalculator.progressCallback = None
        cleanedCalculator.netWrapper = None
        cleanedCalculator.activationCache = None
        cleanedCalculator.imageDataset = None

        workerProcessCount = options.threadCount - 1
        processes = []
        # for i in range(options.threadCount):
        #     # processes.append(multiprocessing.Process(target=CMultiThreadedCalculator.workerThreadFunc,
        #     #                                          args=[cleanedCalculator], name='Process %d' % i))
        #     processes.append(multiprocessing.Process(target=workerProcessFunc2, \
        #                                              args=[cleanedCalculator, 'Process %d' % (i)]))
        pool = multiprocessing.Pool(workerProcessCount)
        # runProcessCount = 0

        # batchSize = calculator.batchSize
        curDataList = []
        for epochNum in epochNums:
            if epochNum < 100:
                continue
            epochActivations = calculator.netWrapper.getImagesActivations_Batch(
                            calculator.layerName, range(1, calculator.imageToProcessCount + 1),
                            epochNum)
            print('Epoch %d activations prepared' % epochNum)
            # if self.threadCrashed:
            #     print("Crashing detected")
            #     break
            # if runProcessCount < options.threadCount:
            #     processes[runProcessCount].start()
            #     runProcessCount += 1
            curDataList.append((cleanedCalculator, epochNum, epochActivations))
            if len(curDataList) == workerProcessCount:
                strInfo = 'Running for epochs to %d, %s in cache' % \
                            (curDataList[-1][1], self.mainCalculator.netWrapper.getCacheStatusInfo())
                if not self.mainCalculator.progressCallback.onEpochProcessed(strInfo):
                    curDataList = []
                    break
                t = threading.Thread(None, pool.map,
                           'Epochs %d, ... map thread' % curDataList[0][1], [workerProcessFunc2, curDataList])
                t.start()
                curDataList = []
            # print('Result for epoch %d received' % epochNum)

            if epochNum % 4 == 0:
                calculator.netWrapper.saveState()
        if curDataList:
             t = threading.Thread(None, pool.map,
                        'Epochs %d, ... map thread' % curDataList[0][1], [workerProcessFunc2, curDataList])
        self.stopEvent.set()
        for p in processes:
            p.join()

        print("Finished")

    @staticmethod
    def workerProcessFunc(calculator, threadName):
        try:
            print("%s: started" % threadName)

            import MnistNetVisWrapper

            calculator.netWrapper = MnistNetVisWrapper.CMnistVisWrapper()
            calculator.activationCache = calculator.netWrapper.activationCache
            calculator.imageDataset = calculator.netWrapper.getImageDataset()
            calculator.progressIndicator = CMultiThreadedCalculator.TProgressIndicator()
            print("%s: inited" % threadName)

            calculator.epochNum = 15
            (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
                # Dangerous - we use method that can call model. But the cache is big so this should not happen
            resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)
            calculator.saveMultActTopsImage(resultImage)

            infoStr = '%s: epoch %d ready, %s in cache, queue %d' % \
                        (threadName, calculator.epochNum, mainObj.mainCalculator.netWrapper.getCacheStatusInfo(),
                         mainObj.activationsQueue.qsize())
            print(infoStr)
        except Exception as ex:
            print("Error in %s workerThreadFunc: %s" % (threadName, str(ex)))
            # mainObj.threadCrashed = True
            # mainObj.stopEvent.set()

        print("%s: finished" % threadName)

    class TProgressIndicator:
        def __init__(self, mainWindow, calculator, threadInfo=None):
            self.mainWindow = mainWindow
            self.calculator = calculator
            self.threadInfo = threadInfo
            # self.mainWindowLock = _thread.allocate_lock()

        def onMultActTopsProgress(self, infoStr, processedImageCount=None, curBestSourceCoords=None):
            print(infoStr)
            return True

    @staticmethod
    def workerThreadFunc(mainObj, threadName):
        print("%s: started" % threadName)
        try:
            calculator = copy.copy(mainObj.mainCalculator)
            cancelling = False
            while (not mainObj.stopEvent.is_set() or not mainObj.activationsQueue.empty()) and \
                    not cancelling:
                (epochNum, epochActivations) = mainObj.activationsQueue.get()
                print("%s: processing epoch %d, queue size %s" %
                      (threadName, epochNum, '*' * mainObj.activationsQueue.qsize()))

                calculator.epochNum = epochNum
                (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
                    # Dangerous - we use method that can call model. But the cache is big so this should not happen
                resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)
                calculator.saveMultActTopsImage(resultImage)

                infoStr = '%s: epoch %d ready, %s in cache, queue %d' % \
                            (threadName, epochNum, mainObj.mainCalculator.netWrapper.getCacheStatusInfo(),
                             mainObj.activationsQueue.qsize())
                        # progressCallback - like QMainWindow.TProgressIndicator

                # with mainObj.mainWindowLock:
                #     cancelling = not mainObj.mainCalculator.progressCallback.onEpochProcessed(infoStr)

        except Exception as ex:
            print("Error in %s workerThreadFunc: %s" % (threadName, str(ex)))
            mainObj.threadCrashed = True
            mainObj.stopEvent.set()

        print("%s: finished" % threadName)



# Multithreaded producer-consumer example from https://realpython.com/intro-to-python-threading/#producer-consumer-threading
import concurrent.futures
import logging
import queue
import random

def producer(queue, event):
    """Pretend we're getting a number from the network."""
    pauseSum = 0
    pauseCount = 0
    if 0:
        x = np.random.normal(0.1, 0.05, size=(1000, 100))
        m = np.mean(x)
        m2 = np.mean(x[x >= 0])
    while not event.is_set():
        message = random.randint(1, 101)
        pause = np.random.normal(0.1, 0.05)
        while pause < 0:
            pause = np.random.normal(0.1, 0.05)
        pauseSum += pause
        pauseCount += 1
        logging.info("Producer message: %s, average pause %.5f s, next %.3f s",
                     message, pauseSum / pauseCount, pause)
        queue.put(message)
        time.sleep(pause)

    logging.info("Producer received event. Exiting")

def consumer(queue, event):
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not queue.empty():
        message = queue.get()
        logging.info(
            "Consumer storing message: %2s, size %s", message, '*' * queue.qsize()
        )
        time.sleep(0.3)

    logging.info("Consumer received event. Exiting")


import multiprocessing

def workerProcessFunc2(params):
    try:
        threadName = 'Process %s' % str(multiprocessing.current_process())
        print("%s: started" % threadName)
        (calculator, epochNum, epochActivations) = params
        setProcessPriorityLow()

        # Overwriting  self.netWrapper.getImagesActivations_Batch(self.layerName, imageNums, self.epochNum),
        # called by calcBestSourceCoords

        # class TBatchDataProvider:
        #     def __init__(self, epochActivations):

        def getBatchData2(_, imageNums, callEpochNum):
            assert callEpochNum == epochNum
            return epochActivations[np.array(imageNums, dtype=int) - 1, :]

        def getBatchData4(_, imageNums, callEpochNum):
            assert callEpochNum == epochNum
            return epochActivations[np.array(imageNums, dtype=int) - 1, :, :, :]

        class TDummy:
            def getObject(self, itemCacheName):
                return None
            def saveObject(self, name, value):
                pass
            def getUsedMemory(self):
                return 0

        import MnistNetVisWrapper

        calculator.netWrapper = MnistNetVisWrapper.CMnistVisWrapper(activationCache=TDummy())
        calculator.netWrapper.getImagesActivations_Batch = \
                getBatchData2 if len(epochActivations.shape) == 2 else getBatchData4
        calculator.activationCache = calculator.netWrapper.activationCache
        calculator.imageDataset = calculator.netWrapper.getImageDataset()
        # calculator.mainWindow = VisQtMain.QtMainWindow()
        calculator.progressCallback = TProgressIndicator2(None, None)
        # calculator.netWrapper.loadCacheState()
        print("%s: inited" % threadName)

        calculator.epochNum = epochNum
        (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
            # Dangerous - we use method that can call model. But the cache is big so this should not happen
        resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)
        calculator.saveMultActTopsImage(resultImage)

        infoStr = '%s: epoch %d ready, %s in cache' % \
                    (threadName, calculator.epochNum, calculator.netWrapper.getCacheStatusInfo())
        print(infoStr)
        return infoStr
    except Exception as ex:
        print("Error in %s workerProcessFunc2: %s" % (threadName, str(ex)))
        # mainObj.threadCrashed = True
        # mainObj.stopEvent.set()
        return 'Error: %s' % str(ex)

    print("%s: finished" % threadName)

class TProgressIndicator2:
    def __init__(self, mainWindow, calculator, threadInfo=None):
        self.mainWindow = mainWindow
        self.calculator = calculator
        self.threadInfo = threadInfo
        # self.mainWindowLock = _thread.allocate_lock()

    def onMultActTopsProgress(self, infoStr, processedImageCount=None, curBestSourceCoords=None):
        print(infoStr)
        return True

if __name__ == "__main__":
    if 0:
        format = "%(asctime)s [%(thread)d]: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
                            datefmt="%H:%M:%S")

        pipeline = queue.Queue(maxsize=20)
        event = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.submit(producer, pipeline, event)
            for _ in range(3):
                executor.submit(consumer, pipeline, event)

            # time.sleep(0.6)
            # logging.info("Main: about to set event")
            # event.set()

    if 1:
        calc = dict()
        workerProcessFunc2(TProgressIndicator2(None, None), 'Test process')


