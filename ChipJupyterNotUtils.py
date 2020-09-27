from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from itertools import islice
# from alexnet_utils import imresize

# # import copy
# import datetime
import matplotlib
# matplotlib.use('AGG')
# matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt4agg import FigureCanvas
#     # +FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
# from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
# # from matplotlib import cm
# import pickle
# import math
# import numpy as np
# import PyQt4.Qt
# from PyQt4 import QtCore, QtGui
# # from PySide import QtGui
# # import re
#     # from glumpy import app, glm   # In case of "GLFW not found" copy glfw3.dll to C:\Python27\Lib\site-packages\PyQt4\
#     # from glumpy.transforms import Position, Trackball, Viewport
#     # from glumpy.api.matplotlib import *
#     # import glumpy.app.window.key as keys
import os
# # import random
import sys
# # import time
# import _thread
try:
    from scipy.misc import imsave
except:
    from imageio import imsave  # For scipy >= 1.2, but imresize needs additional searching


# import AlexNetVisWrapper
# import ImageNetsVisWrappers
# import ImagesPyTorchVisWrapper
# import MnistNetVisWrapper
# import MultActTops
# import DataCache
from MyUtils import *
from VisUtils import *

if __name__ != "__main__":
    sys.path.append("src")


def showWeights(weights, colCount):
        # print(weights.shape, weights.min(), weights.max())
        weightsImageData = weights.transpose((1, 2, 3, 0))
        weightsImageData = layoutLayersToOneImage(weightsImageData, colCount, 1, weights.min())
        if 0:
            weightsImageData = weightsImageData / \
                    (weightsImageData.max() - weightsImageData.min()) * 255

            # print(weightsImageData.shape)
            # ax.clear()
            # ax.imshow(weightsImageData, cmap='plasma')
            fig = plt.figure(figsize=(4, 7))
            ax = fig.add_subplot(111)
            ax.imshow(weightsImageData)
        else:
            weightsImageData = np.mean(weightsImageData, axis=-1)
            if weightsImageData.shape[1] > 300:
                fig = plt.figure(figsize=(20, 10))
            else:
                fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111)
            ax.imshow(weightsImageData, cmap='rainbow')
        return fig, weightsImageData

# def printWeightsStats(model):
#     info = []
#     for layerName in model.getAllLayers():
#         weights = model.getMultWeights(layerName)
#         infoStr = '%s: %s, %.4g, %.4g' % (layerName, str(weights.shape), weights.min(), weights.max())
#         print(infoStr)
#         info.append(infoStr)
#     return info

# # Part of QtMainWindow without GUI
# class NetControlObject():
#     c_channelMargin = 2
#     c_channelMargin_Top = 5
#     AllEpochs = -2
#
#     def __init__(self, parent=None):
#         self.exiting = False
#         self.cancelling = False
#         self.lastAction = None
#         self.lastActionStartTime = None
#         # self.netWrapper = AlexNetVisWrapper.CAlexNetVisWrapper()
#         # self.netWrapper = ImageNetsVisWrappers.CImageNetVisWrapper()
#         self.netWrapper = ImagesPyTorchVisWrapper.CPyTorchImageNetVisWrapper()
#         # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper()
#         # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper3_Towers()
#         # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper4_Matrix()
#         # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper5_DeeperTowers()
#         self.activationCache = self.netWrapper.activationCache
#         self.imageDataset = self.netWrapper.getImageDataset()
#         self.tensorFlowLock = threading.Lock()
#         # self.savedNetEpochs = None
#         # self.curEpochNum = 0
#         self.weightsBeforeReinit = None
#         self.weightsReinitInds = None
#         self.weightsReinitEpochNum = None
#
#         self.maxAnalyzeChanCount = 500
#
#     def init(self):
#         # DeepMain.MainWrapper.__init__(self, DeepOptions.studyType)
#         # DeepMain.MainWrapper.init(self)
#         # DeepMain.MainWrapper.startNeuralNetTraining()
#         # self.net = self.netTrader.net
#         # self.fastInit()
#
#         if not os.path.exists('Data/BestActs'):
#             os.makedirs('Data/BestActs')
#         if not os.path.exists('Data/Correlations'):
#             os.makedirs('Data/Correlations')
#         # os.makedirs('Data/Weights')
#         self.loadNetStateList()
#         self.curEpochNum = self.savedNetEpochs[-1]
#         if str(self.curEpochNum).find('--') >= 0:
#             self.curEpochNum = NetControlObject.AllEpochs
#         # self.epochComboBox.setCurrentIndex(self.epochComboBox.count() - 1)
#
#         self.curBlockInd = 1
#         # self.blockComboBox.setCurrentIndex(1)
#         self.curImageNum = 200
#         self.curChanNum = 0
#         self.learnRate = 0.0001
#
#         self.figure = figure(figsize=(9, 5))
#         # self.figure2 = figure(figsize=(6, 4))
#         # self.figure = None
#         self.gridSpec = matplotlib.gridspec.GridSpec(2,2, width_ratios=[1,3], height_ratios=[1,1])
#
#     def showProgress(self, str, processEvents=True):
#         print(str)
#         # # self.setWindowTitle(str)
#         # self.infoLabel.setText(str)
#         # if processEvents:
#         #     PyQt4.Qt.QCoreApplication.processEvents()
#
#     def startAction(self, actionFunc):
#         self.lastAction = actionFunc
#         self.lastActionStartTime = datetime.datetime.now()
#         self.cancelling = False
#
#     def getSelectedEpochNum(self):
#         return self.curEpochNum
#
#     def getPrevLayerName(self, layerName):
#         import re
#
#         result = re.search(r'conv_(\d+)(.*)', layerName)
#         if result:
#             layerNum = int(result.group(1))
#             return 'conv_%d%s' % (layerNum - 1, result.group(2))
#         else:
#             raise Exception('Can\'t decode layer number')
#
#     def getSelectedImageNum(self):
#         return self.curImageNum
#         # return self.imageNumEdit.value()
#             # int(self.imageNumLineEdit.text())
#
#     def getSelectedLayerName(self):
#         return self.curLayerName
#
#     def getLearnRate(self):
#         return self.learnRate
#
#     def getSelectedChannelNum(self):
#         return self.curChanNum
#         # return self.chanNumEdit.value()
#
#     def getChannelsToAnalyze(self, data):
#         if self.maxAnalyzeChanCount and data.shape[0] > self.maxAnalyzeChanCount:
#             return data[:self.maxAnalyzeChanCount]
#         else:
#             return data
#
#     def getChannelToAnalyzeCount(self, data):
#         chanCount = data.shape[0]
#         if self.maxAnalyzeChanCount and chanCount > self.maxAnalyzeChanCount:
#             chanCount = self.maxAnalyzeChanCount
#         colCount = math.ceil(math.sqrt(chanCount) * 1.15 / 2) * 2
#         if colCount in [4, 6] and chanCount % (colCount - 1) == 0:
#             colCount -= 1
#         chanCount = chanCount // colCount * colCount
#         return (chanCount, colCount)
#
#
#     def closeEvent(self, event):
#         self.exiting = True
#         self.netWrapper.cancelling = True
#         print("Close event")
#
#     def onCancelPressed(self):
#         # self.lastAction = None
#         self.cancelling = True
#         self.netWrapper.cancelling = True
#         self.showProgress('Cancelling...')
#
#     def onShowImagePressed(self):
#         self.startAction(self.onShowImagePressed)
#         imageNum = self.getSelectedImageNum()
#         imageData = self.imageDataset.getImage(imageNum, 'cropped')
#         # mi = imageData.min()
#         # ma = imageData.max()
#         # for i in range(0, 255, 2):
#         #     imageData[i // 2, 50:100, :] = [i, 0, 0]
#         #     imageData[i // 2, 105:160, :] = [0, i, 0]
#         #     imageData[i // 2, 180:220, :] = [0, 0, i]
#         # imageData[:, [1, 4, 7], :] = [[255, 0, 0], [0, 200 ,0], [0, 0, 145]]
#
#         display(imageData)
#         # ax = self.figure.add_subplot(self.gridSpec[0, 0])       # GridSpec: [y, x]
#         # self.showImage(ax, imageData)
#         # # ax.imshow(imageData, extent=(-100, 127, -100, 127), aspect='equal')
#         # self.drawFigure()
#
#     def clearFigure(self):
#         # In GUI application clearFigure means that we want to reuse screen, in Jupyter notebook -
#         # that we want to display a new figure
#         plt.clf()
#         print('clf')
#         # self.figure = figure(figsize=(8, 5))
#         # self.figure.clear()
#         self.mainSubplotAxes = None
#         # self.figure.set_tight_layout(True)
#
#     def getMainSubplot(self):
#         self.figure = figure(figsize=(8, 5))
#         # print('add_subplot')
#         return self.figure.add_subplot()
#         # if not hasattr(self, 'mainSubplotAxes') or self.mainSubplotAxes is None:
#         #     self.mainSubplotAxes = self.figure.add_subplot(self.gridSpec[:, 1])
#         # return self.mainSubplotAxes
#
#     def showImage(self, ax, imageData):
#         # ax.clear()    # Including clear here is handy, but not obvious
#         # print('Showing image ', imageData.shape)
#         if len(imageData.shape) >= 3:
#             if imageData.shape[2] > 1:
#                 return ax.imshow(imageData.astype(np.uint8)) # , aspect='equal')
#             else:
#                 imageData = np.squeeze(imageData, axis=2)
#                 # if imageData.dtype == np.float32:
#                 #     imageData *= 255
#                 return ax.imshow(imageData, cmap='Greys_r')
#         else:
#             return ax.imshow(imageData, cmap='Greys_r')
#
#     def drawFigure(self):
#         # self.canvas.draw()
#         # print('show')
#         plt.show()
#         # self.figure = figure(figsize=(5, 5))
#
#     def onShowActivationsPressed(self, augment=False):
#         self.startAction(self.onShowActivationsPressed)
#         epochNum = self.getSelectedEpochNum()
#         imageNum = self.getSelectedImageNum()
#         layerName = self.getSelectedLayerName()
#
#         activations, drawMode, stdData = self.getActivationsData(epochNum, imageNum, layerName, augment)
#
#         if epochNum == self.AllEpochs:
#             self.clearFigure()
#             ax = self.getMainSubplot()
#
#             im = ax.imshow(activations.transpose(), cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#
#             if not stdData is None:
#                 ax = self.figure.add_subplot(self.gridSpec[1, 0])
#                 self.showImage(ax, stdData.transpose())
#         else:
#             # self.figure.set_tight_layout(True)     # self.figure in this class is initialized in getMainSubplot
#             ax = self.getMainSubplot()
#             ax.clear()
#
#             # plt.subplots_adjust(left=0.01, right=data.shape[0], bottom=0.1, top=0.9)
#             if drawMode == 'map':
#                 colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
#                 data = layoutLayersToOneImage(np.sqrt(activations),
#                                               colCount, margin)
#                 ax.imshow(data, cmap='plasma')
#             else:
#                 ax.plot(activations)
#                 if ax.get_ylim()[0] > ax.get_ylim()[1]:
#                     ax.invert_yaxis()
#
#         self.drawFigure()
#
#     def getActivationsData(self, epochNum, imageNum, layerName, augment=False):
#         if epochNum == self.AllEpochs:
#             epochNums = self.netWrapper.getSavedNetEpochs()
#             if len(epochNums) == 0:
#                 self.showProgress('Activations error: no saved epochs')
#                 epochNum = -10
#
#         if epochNum == self.AllEpochs:
#             dataList = []
#             t0 = datetime.datetime.now()
#
#             for curEpochNum in self.netWrapper.getSavedNetEpochs():
#                 dataList.append(self.netWrapper.getImagesActivations_Batch(
#                         layerName, [imageNum], curEpochNum, augment=augment)[0])
#                 t = datetime.datetime.now()
#                 if (t - t0).total_seconds() >= 1:
#                     self.showProgress('Analyzed epoch %d' % curEpochNum)
#                     t0 = t
#                     if self.cancelling or self.exiting:
#                         self.showProgress('Cancelled')
#                         break
#             activations = np.stack(dataList, axis=0)
#             self.showProgress('Activations: %s, min %.4f, max %.4f (%s)' % \
#                     (str(activations.shape), activations.min(), activations.max(),
#                      str([int(v[0]) for v in np.where(activations == activations.max())])))
#
#             drawMode = 'map'
#             stdData = None
#             if len(activations.shape) > 2:
#                 axisToStick = tuple(range(2, len(activations.shape)))
#                 stdData = np.std(activations, axis=axisToStick)
#                 activations = np.abs(activations)
#                 activations = np.mean(activations, axis=axisToStick)
#         else:
#             activations = self.netWrapper.getImagesActivations_Batch(
#                     layerName, [imageNum], epochNum, augment=augment)
#             self.showProgress('Activations: %s, min %.4f, max %.4f (%s)' % \
#                     (str(activations.shape), activations.min(), activations.max(),
#                      str([int(v[0]) for v in np.where(activations == activations.max())])))
#
#             drawMode = 'map'
#             stdData = None
#             if len(activations.shape) == 2:   # Dense level scalars
#                 if activations.shape[1] < 50:
#                     drawMode = 'plot'
#                     activations = activations.flatten()
#                 else:
#                     activations = np.reshape(activations, [activations.shape[1], 1, 1])
#                     margin = 0
#             else:
#                 activations = self.getChannelsToAnalyze(activations[0])
#                 margin = self.c_channelMargin
#         return activations, drawMode, stdData
#
#     def onShowActEstByImagesPressed(self):
#         self.startAction(self.onShowActEstByImagesPressed)
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = max(100, self.getSelectedImageNum())
#         layerName = self.getSelectedLayerName()
#
#         imagesActs = self.netWrapper.getImagesActivations_Batch(
#                 layerName, range(1, firstImageCount + 1), epochNum)
#         self.showProgress('Activations: %s, max %.4f (%s)' % \
#                 (str(imagesActs.shape), imagesActs.max(),
#                  str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
#         ests = self.getEstimations(imagesActs)
#
#         self.clearFigure()
#         # ax = self.getMainSubplot()
#
#         # ax = self.figure.add_subplot(self.gridSpec[0, 0])
#
#         if 1:
#             (ests, data2) = self.sortEstimations(ests, imagesActs)
#         else:
#             data2 = imagesActs
#         data2 = data2.transpose([1, 0] + list(range(2, len(imagesActs.shape))))
#         self.showGradients(ests.transpose([1, 0]), data2, False)
#         # self.showGradients(ests, imagesActs, True)  # Transpose will be made inside, but also undesirable log10
#
#     def onShowImagesWithTSnePressed(self):
#         self.startAction(self.onShowImagesWithTSnePressed)
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = max(300, self.getSelectedImageNum())
#         layerName = self.getSelectedLayerName()
#
#         imagesActs = self.netWrapper.getImagesActivations_Batch(
#                 layerName, range(1, firstImageCount + 1), epochNum)
#         self.showProgress('Activations: %s, max %.4f (%s)' % \
#                 (str(imagesActs.shape), imagesActs.max(),
#                  str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
#         ests = self.getEstimations(imagesActs)
#
#         if 1:
#             self.showTSneByImages(ests, firstImageCount)
#         else:
#             self.showTSneByChannels(ests)
#
#     def showTSneByImages(self, ests, firstImageCount):
#         from sklearn.manifold import TSNE
#
#         tsne = TSNE(n_components=2, verbose=9, perplexity=40, n_iter=250, random_state=1)
#         # for iterCount in range(250, 2000000, 50):
#         iterCount = 3000
#         while iterCount < 6000000:
#             tsne.n_iter = iterCount
#             tsne.n_iter_without_progress = 250
#             tsneCoords = tsne.fit_transform(ests)
#             self.showProgress('T-SNE: %d iterations made' % iterCount)
#
#             ax = self.getMainSubplot()
#             ax.clear()
#             (extent, imageSize) = self.getTsneResultsExtent(tsneCoords)
#             selectedImageInds = getMostDistantPoints(tsneCoords, tsneCoords.shape[0] // 4)
#             np.savetxt('Data/DistantImageNums1.txt', selectedImageInds[0], fmt='%d', delimiter='')
#             np.savetxt('Data/DistantImageNums2.txt', selectedImageInds[1], fmt='%d', delimiter='')
#
#             for i, coords in enumerate(tsneCoords):
#                 imageNum = i + 1
#                 imageData = self.imageDataset.getImage(imageNum, 'cropped')
#                 imageData = np.squeeze(imageData, axis=2)
#                 ax.imshow(imageData, cmap='Greys_r', alpha=0.05,
#                           extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))
#
#             for imageInd in selectedImageInds[0]:
#                 imageNum = imageInd + 1
#                 imageData = self.imageDataset.getImage(imageNum, 'cropped')
#                 imageData = np.squeeze(imageData, axis=2)
#                 coords = tsneCoords[imageInd]
#                 ax.imshow(imageData, cmap='Greys_r', alpha=0.35,
#                           extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))
#             for imageInd in selectedImageInds[1]:
#                 imageNum = imageInd + 1
#                 imageData = self.imageDataset.getImage(imageNum, 'cropped')
#                 imageData = np.squeeze(imageData, axis=2)
#                 coords = tsneCoords[imageInd]
#                 ax.imshow(imageData, cmap='plasma', alpha=0.35,
#                           extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))
#
#             ax.imshow(imageData, cmap='Greys_r', alpha=0,
#                       extent=extent)
#
#             ax = self.figure.add_subplot(self.gridSpec[0, 0])
#             ax.clear()
#             labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
#             ax.scatter(tsneCoords[:, 0], tsneCoords[:, 1], c=labels, alpha=0.6, cmap='plasma')
#
#             self.drawFigure()
#             self.showProgress('T-SNE: %d iterations displayed' % iterCount)
#
#             if self.cancelling or self.exiting:
#                 return
#             iterCount = int(iterCount * 1.2)
#             break
#             # iterCount += 50
#
#     def showTSneByChannels(self, ests):
#         from sklearn.manifold import TSNE
#
#         self.needShowCurMultActTops = False
#         calculator = self.fillMainMultActTopsOptions()
#         calculator.chanCount = ests.shape[1]
#         if not calculator.checkMultActTopsInCache():
#             (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
#         else:
#             (bestSourceCoords, processedImageCount) = (None, None)
#         chanImages = calculator.prepareMultActTopsImageData(bestSourceCoords, processedImageCount)
#         chanImages = np.squeeze(chanImages, axis=3)
#         # resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)
#
#         tsne = TSNE(n_components=2, verbose=9, perplexity=30, n_iter=250, random_state=1)
#         # for iterCount in range(250, 2000000, 50):
#         iterCount = 250
#         assert len(ests.shape) == 2
#         tsneData = np.transpose(ests)
#         while iterCount < 1000000:
#             tsne.n_iter = iterCount
#             tsne.n_iter_without_progress = 250
#             tsneCoords = tsne.fit_transform(tsneData)
#             self.showProgress('T-SNE: %d iterations made' % iterCount)
#
#             ax = self.getMainSubplot()
#             ax.clear()
#             (extent, imageSize) = self.getTsneResultsExtent(tsneCoords)
#             for chanInd, coords in enumerate(tsneCoords):
#                 imageData = chanImages[chanInd]
#                 ax.imshow(imageData, cmap='Greys_r', alpha=0.5,
#                           extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))
#
#             ax.imshow(imageData, cmap='Greys_r', alpha=0,
#                       extent=extent)
#                     # (tsneCoords[:, 0].min(), tsneCoords[:, 0].max() + imageSize,
#                     #  tsneCoords[:, 1].min(), tsneCoords[:, 1].max() + imageSize))
#
#             # ax = self.figure.add_subplot(self.gridSpec[0, 0])
#             # ax.clear()
#             # labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
#             # ax.scatter(tsneCoords[:, 0], tsneCoords[:, 1], c=labels, alpha=0.6, cmap='plasma')
#
#             self.drawFigure()
#             self.showProgress('T-SNE: %d iterations displayed' % iterCount)
#             if self.cancelling or self.exiting:
#                 return
#             # iterCount = int(iterCount * 1.25)
#             iterCount += 50
#
#     def getTsneResultsExtent(self, tsneCoords):
#         c_extentSkipCount = 5
#         c_maxEnlargeMult = 0.1
#
#         # Skipping first most distant (min and max x and y) points
#         # and then returning those of them which are not too far
#         sortedXs = np.sort(tsneCoords[:, 0])
#         sortedYs = np.sort(tsneCoords[:, 1])
#         extentWidth = sortedXs[-c_extentSkipCount] - sortedXs[c_extentSkipCount]
#         extentHeight = sortedYs[-c_extentSkipCount] - sortedYs[c_extentSkipCount]
#         # extent = (sortedXs[c_extentSkipCount], sortedXs[-c_extentSkipCount],
#         #           sortedYs[c_extentSkipCount], sortedYs[-c_extentSkipCount])
#         imageSize = max(extentWidth, extentHeight) / \
#                 math.sqrt(tsneCoords.shape[0]) / 2
#         extent = [sortedXs[c_extentSkipCount], sortedXs[-c_extentSkipCount],
#                   sortedYs[c_extentSkipCount], sortedYs[-c_extentSkipCount]]
#         for i in range(c_extentSkipCount - 1, -1, -1):
#             if extent[0] - extentWidth * c_maxEnlargeMult <= sortedXs[i]:
#                 extent[0] = sortedXs[i]
#             if extent[1] + extentWidth * c_maxEnlargeMult >= sortedXs[-i - 1]:
#                 extent[1] = sortedXs[-i - 1]
#             if extent[2] - extentWidth * c_maxEnlargeMult <= sortedYs[i]:
#                 extent[2] = sortedYs[i]
#             if extent[3] + extentWidth * c_maxEnlargeMult >= sortedYs[-i - 1]:
#                 extent[3] = sortedYs[-i - 1]
#         extent[1] += imageSize
#         extent[3] += imageSize
#         # center = ((sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount]) / 2,
#         #           (sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount]) / 2,
#         # extent = ((sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount])
#         return (extent, imageSize)
#
#     def getEstimations(self, imagesActs):
#         shape = imagesActs.shape
#         if len(shape) == 2:
#             return imagesActs
#         else:
#             data = abs(imagesActs)
#             # counts = np.zeros(shape)
#             # counts[data >= 0.1] = 1
#             # counts = np.sum(counts, axis=tuple(range(2, len(shape))))
#             mean = np.mean(data)
#             # boundary = 0.2
#             boundary = mean / 1.8
#             counts = np.count_nonzero(data >= boundary, axis=tuple(range(2, len(shape))))
#
#             means = np.mean(imagesActs, axis=tuple(range(2, len(shape))))
#             return np.concatenate([counts, means], axis=1)
#
#     def sortEstimations(self, ests, imagesActs):
#         # First dimension here is images
#
#         if 0:
#             # Sorting by sum of estimations
#             sum = np.sum(ests, axis=1)
#             newInds = sum.argsort()
#             ests = ests[newInds, :]
#             # imagesActs = ests[np.sum(imagesActs, axis=1).argsort(), :]
#             imagesActs = imagesActs[newInds, :]
#         else:
#             firstImageCount = ests.shape[0]
#             labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
#             mean = np.mean(ests, axis=1)
#             mean -= mean.min()
#             labels += mean / (mean.max() + 1e-15) / 2
#             newInds = labels.argsort()
#             ests = ests[newInds]
#             imagesActs = imagesActs[newInds]
#
#         return (ests, imagesActs)
#
#     def onShowActTopsPressed(self):       # It would be desirable to move into MultActTops, but this requires time and the code is not necessary
#         self.startAction(self.onShowActTopsPressed)
#         imageNum = self.getSelectedImageNum()
#         sourceImageData = self.imageDataset.getImage(imageNum, 'cropped')
#         # alexNetImageData = self.imageDataset.getImage(imageNum, 'net')
#         layerName = self.getSelectedLayerName()
#         activations = self.netWrapper.getImageActivations(layerName, imageNum)
#         activations = self.getChannelsToAnalyze(activations[0])
#         if len(activations.shape) == 1:
#             return
#             # activations = np.expand_dims(activations, axis=0)
#
#         sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(layerName)
#         if sourceBlockCalcFunc is None:
#             return
#         colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
#         resultList = []
#         # if layerName in ['conv_1', 'conv_2']
#         #     activations[0, 22:25, 0] = 100     #d_
#         for chanInd in range(activations.shape[0]):
#             vals = attachCoordinates(activations[chanInd])   # Getting list of (x, y, value)
#             sortedVals = vals[:, vals[2, :].argsort()]
#             sourceBlock = sourceBlockCalcFunc(int(sortedVals[0, -1]), int(sortedVals[1, -1]))
#             resultList.append(sourceImageData[sourceBlock[0] : sourceBlock[2], sourceBlock[1] : sourceBlock[3], :])
#             # sourceUpperLeft = (int(sortedVals[0, -1]) * mult, int(sortedVals[1, -1]) * mult)
#             # resultList.append(sourceImageData[sourceUpperLeft[1] : sourceUpperLeft[1] + size,
#             #                                   sourceUpperLeft[0] : sourceUpperLeft[0] + size, :])
#
#             # selectedList = []
#             # i = vals.shape[0] - 1
#             # while len(selectedList) < 9:
#             #     curVal = vals[i]
#             #     found = False
#             #     for prevVal in selectedList
#         # maxSize = 0
#         # for result in resultList:
#         #     if maxSize < max(result.shape[0:1]):
#         #         maxSize = max(result.shape[0:1])
#         # resultList2 = []
#         # for result in resultList:
#         #     if maxSize != result.shape[0] or maxSize != result.shape[1]:
#         #         padded = np.pad(result, ((0, maxSize - result.shape[0]),
#         #                                  (0, maxSize - result.shape[1]), (0, 0)),
#         #                         constant_values=255)
#         #     else:
#         #         padded = result
#         #     resultList2.append(padded)
#         resultList = padImagesToMax(resultList)
#         data = np.stack(resultList, axis=0)
#         data = layoutLayersToOneImage(data, colCount, self.c_channelMargin)
#
#         ax = self.figure.add_subplot(self.gridSpec[1, 0])
#         ax.clear()
#         self.showImage(ax, data)
#         self.drawFigure()
#
#     def onShowActTopsFromCsvPressed(self):
#         # Fast, based on data produced by activations.py
#         self.startAction(self.onShowActTopsFromCsvPressed)
#         calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
#         options = calculator
#         options.layerNum = self.blockComboBox.currentIndex() + 1
#         calculator.showActTops_FromCsv()
#
#
#     def fillMainMultActTopsOptions(self):
#         calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
#         options = calculator
#         options.epochNum = self.getSelectedEpochNum()
#         if options.epochNum is None:
#             epochNums = self.netWrapper.getSavedNetEpochs()
#             options.epochNum = epochNums[-1]
#         layerName = self.getSelectedLayerName()
#         options.layerName = layerName
#         # options.embedImageNums = True
#         options.imageToProcessCount = max(100 if self.netWrapper.name == 'mnist' else 20, \
#                     self.getSelectedImageNum())
#
#         # Getting activations to get channel count. They should be reread from cache later
#         # try:
#         activations1 = self.netWrapper.getImageActivations(
#                             layerName, 1, options.epochNum)
#         # except Exception as ex:
#         #     self.showProgress("Error: %s. Looking for children layers" % str(ex))
#         #     activations1 = self.net.getCombinedLayerImageActivations(layerName, 1, options.epochNum)
#         activations1 = self.getChannelsToAnalyze(activations1[0])
#         (options.chanCount, options.colCount) = self.getChannelToAnalyzeCount(activations1)
#         if layerName.find('reshape') >= 0:
#             options.colCount = 32
#         print('options.chanCount', options.chanCount)
#
#         calculator.progressCallback = NetControlObject.TProgressIndicator(self, calculator)
#         return calculator
#
#     def buildMultActTops(self):
#         # My own implementation, from scratch, with images subblocks precision.
#         # Drawing of images plot from Python is very slow in JupyterLab for some reason.
#         # So we only prepare it here and calling imshow in the notebook
#         self.startAction(self.buildMultActTops)
#         calculator = self.fillMainMultActTopsOptions()
#         if calculator.checkMultActTopsInCache():
#             # No need to collect activations, everything will be taken from cache
#             # resultImage = calculator.showMultActTops(None)
#             resultImage = calculator.buildMultActTopsImage(None)
#
#             # calculator.saveMultActTopsImage(resultImage)
#             return resultImage
#
#         self.needShowCurMultActTops = False
#         # self.multActTopsButton.setText('Save current')
#         # try:
#         #     self.multActTopsButton.clicked.disconnect()
#         # except e:
#         #     pass
#         # self.multActTopsButton.clicked.connect(self.onShowCurMultActTopsPressed)
#
#         # activations = self.getChannelsToAnalyze(self.netWrapper.getImageActivations(
#         #           layerNum, 1, options.epochNum)[0])
#         # print(activations)
#
#         # try:
#         (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
#         # finally:
#         #     self.multActTopsButton.setText(self.multActTopsButtonText)
#         #     self.multActTopsButton.clicked.disconnect()
#         #     self.multActTopsButton.clicked.connect(self.onShowMultActTopsPressed)
#
#         if not self.exiting:
#             # resultImage = calculator.showMultActTops(bestSourceCoords, processedImageCount)
#             resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)
#             print(resultImage.shape)
#             if resultImage.shape[0] > 10000:
#                 print('New shape: ', (resultImage.shape[1] // 3, resultImage.shape[0] // 3))
#                 resultImage = imresize(resultImage,
#                         (resultImage.shape[1] // 3, resultImage.shape[0] // 3))
#
#             calculator.saveMultActTopsImage(resultImage, processedImageCount)
#         return resultImage
#
#     class TProgressIndicator:
#         def __init__(self, mainWindow, calculator, threadInfo=None):
#             self.mainWindow = mainWindow
#             self.calculator = calculator
#             self.threadInfo = threadInfo
#             # self.mainWindowLock = _thread.allocate_lock()
#
#         # Returns false if the process should be stopped.
#         # Can show and save intermediate results. When called from showMultActTops this is not necessary
#         # since they already were called and the image is about to be prepared, showed and/or saved
#         def onMultActTopsProgress(self, infoStr, processedImageCount=None, curBestSourceCoords=None):
#             if not self.threadInfo is None:
#                 infoStr = '%s: %s' % (self.threadInfo, infoStr)
#            # with self.mainWindowLock:
#             self.mainWindow.showProgress(infoStr)
#                 # Usually doesn't display anything from other threads
#
#             if self.mainWindow.cancelling or self.mainWindow.exiting:
#                 return False
#             elif self.mainWindow.needShowCurMultActTops and not curBestSourceCoords is None:
#                 self.mainWindow.needShowCurMultActTops = False
#                 resultImage = self.calculator.showMultActTops(curBestSourceCoords, processedImageCount)
#                 self.calculator.saveMultActTopsImage(resultImage, processedImageCount)
#             return True
#
#         def onEpochProcessed(self, infoStr):
#             self.mainWindow.showProgress(infoStr)
#             if self.mainWindow.cancelling or self.mainWindow.exiting:
#                 return False
#             return True
#
#
#     def buildMultActTops_MultiProcess(self, multiOptions):
#         self.startAction(self.buildMultActTops_MultiProcess)
#         mtCalculator = MultActTops.CMultiThreadedCalculator()
#         mtCalculator.run_MultiProcess_MultiOptions(multiOptions)
#
#     def buildMultActTops_MultiThreaded_AllEpochs(self):
#         self.startAction(self.buildMultActTops_MultiThreaded_AllEpochs)
#         calculator = self.fillMainMultActTopsOptions()
#         self.needShowCurMultActTops = False
#         epochNums = self.netWrapper.getSavedNetEpochs()
#         calculator.epochNum = None
#         calculator.threadCount = min(20, getCpuCoreCount() * 2)
#         mtCalculator = MultActTops.CMultiThreadedCalculator()
#
#         # mtCalculator.run(calculator, epochNums)
#         mtCalculator.run_MultiProcess(calculator, epochNums)
#
#     def calcMultActTops_MultipleTfThreaded(self):
#         self.startAction(self.calcMultActTops_MultiThreaded)
#         calculator = self.fillMainMultActTopsOptions()
#         epochNums = self.netWrapper.getSavedNetEpochs()
#         calculator.epochNum = None
#
#         import tensorflow as tf
#
#         # All these graph tricks don't help - it throws exception from model.predict
#         # "'_thread._local' object has no attribute 'value'.
#         # And they consume a lot of memory for some reason (4 GB for 4 wrappers in total)
#         self.netWrappers = []
#         for _ in range(3):
#             threadGraph = tf.Graph()
#
#             with threadGraph.as_default():
#                 threadSession = tf.compat.v1.Session()  #  tf.Session()
#                 with threadSession.as_default():
#                     netWrapper = MnistNetVisWrapper.CMnistVisWrapper( \
#                            self.imageDataset, self.activationCache)
#                     netWrapper._getNet(options.layerName)
#                     netWrapper.threadGraph = threadGraph
#                     netWrapper.threadSession = threadSession
#                     # netWrapper.net.model._make_predict_function()
#             # threadGraph.finalize()
#             threadGraph.switch_to_thread_local()
#             self.netWrappers.append(netWrapper)
#         # tf.compat.v1.get_default_graph().finalize()
#
#         options.threadCount = 3
#         # threads = []
#         calc_MultiThreaded(options.threadCount, self.calcMultActTops_MultipleTfThreadFunc,
#                            epochNums, calculator)
#         self.showProgress('%d threads finished' % options.threadCount)
#
#     def calcMultActTops_MultipleTfThreadFunc(self, threadParams, options):
#         downloadStats = threadParams.downloadStats
#         self.showProgress('%s: started for %s%s' % \
#               (threadParams.threadName, threadParams.ids[:20], \
#                '...' if len(threadParams.ids) > 20 else ''))
#
#         curThreadAdded = 0
#         try:
#             # t0 = datetime.datetime.now()
#             # calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
#             epochNum = -1
#
#             # import tensorflow as tf
#
#             calculator = copy.copy(options)
#             # with self.tensorFlowLock:
#             #     g = tf.Graph()
#             #     with g.as_default():
#             #         calculator.netWrapper = MnistNetVisWrapper.CMnistVisWrapper(
#             #                 self.imageDataset, self.activationCache)
#             #         net = calculator.netWrapper._getNet()
#             #         calculator.netWrapper.model._make_predict_function()
#             #     g.finalize()
#             #     print('%s: net inited' % threadParams.threadName)
#
#             calculator.netWrapper = self.netWrappers[threadParams.ids[0] % 3]
#             calculator.progressCallback = NetControlObject.TProgressIndicator(
#                     self, calculator, threadParams.threadName)
#             with calculator.netWrapper.threadGraph.as_default():
#                 with calculator.netWrapper.threadSession.as_default():
#                     for epochNum in threadParams.ids:
#                         calculator.epochNum = epochNum
#                         (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
#                         if self.cancelling or self.exiting:
#                             break
#                         resultImage = calculator.showMultActTops(bestSourceCoords, processedImageCount)
#                         calculator.saveMultActTopsImage(resultImage)
#                         self.showProgress('%s: epoch %d done' % (threadParams.threadName, epochNum))
#         except Exception as errtxt:
#             print('Exception at %s on epoch %d: %s' % (threadParams.threadName, epochNum, errtxt))
#
#         with downloadStats.updateLock:
#             downloadStats.finishedThreadCount += 1
#
#
#     def onShowCurMultActTopsPressed(self):
#         self.needShowCurMultActTops = True
#
#     def onShowChanActivationsPressed(self):
#         try:
#             self.startAction(self.onShowChanActivationsPressed)
#             epochNum = self.getSelectedEpochNum()
#             imageNum = self.getSelectedImageNum()
#             layerName = self.getSelectedLayerName()
#             chanNum = self.getSelectedChannelNum()
#             activations = self.netWrapper.getImageActivations(layerName, imageNum, epochNum)
#             activations = activations[0][chanNum]
#
#             self.clearFigure()
#             showCurLayerActivations = True
#             if showCurLayerActivations:
#                 imageData = np.sqrt(activations)
#                 ax = self.figure.add_subplot(self.gridSpec[0, 0])
#                 ax.clear()
#                 ax.imshow(imageData, cmap='plasma')
#
#             vals = attachCoordinates(activations)        # Getting list of (x, y, value)
#             sortedVals = vals[:, vals[2, :].argsort()]
#             curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))
#
#             # ax = self.figure.add_subplot(self.gridSpec[0, 0])
#             # ax.clear()
#             # ax.hist([1,2,3], bins=100)
#
#             # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
#             # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
#             allWeights = self.netWrapper.getMultWeights(layerName)
#             weights = allWeights[:, :, :, chanNum]
#             if layerName[:6] == 'conv_1':
#                 prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
#             else:
#                 prevLayerName = self.getPrevLayerName(layerName)
#                 prevLayerActivations = self.netWrapper.getImageActivations(prevLayerName, imageNum, epochNum)[0]
#                         # Conv2 - 256 * 27 * 27
#             self.showProgress('Prev. layer act.: %s, max %.4f (%s)' % \
#                 (str(prevLayerActivations.shape), prevLayerActivations.max(),
#                  str([int(v[0]) for v in np.where(prevLayerActivations == prevLayerActivations.max())])))
#
#             if not showCurLayerActivations:
#                 ax = self.figure.add_subplot(self.gridSpec[0, 0])
#                 ax.clear()
#                 ax.hist(weights.flatten(), bins=100)
#                 if ax.get_ylim()[0] > ax.get_ylim()[1]:
#                     ax.invert_yaxis()
#                 ax.set_aspect('auto')
#
#             colCount = math.ceil(math.sqrt(weights.shape[2]) * 1.15 / 2) * 2
#             self.showWeights(self.figure.add_subplot(self.gridSpec[1, 0]), weights, colCount)
#
#             # bias = allWeights[1].numpy()[chanNum]
#             chanConvData = conv2D_BeforeSumming(prevLayerActivations, weights)
#             convImageData = layoutLayersToOneImage(chanConvData, colCount, 1, chanConvData.min())
#             ax = self.getMainSubplot()
#             # ax = self.figure.add_subplot(self.gridSpec[1, 0])
#             # ax = self.figure.add_subplot(self.gridSpec[:, 1])
#             ax.clear()
#             im = ax.imshow(convImageData, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#             # self.figure.get_colorbar(ax=ax)
#             self.drawFigure()
#         except Exception as ex:
#             self.showProgress("Error: %s" % str(ex))
#
#     def onShowSortedChanActivationsPressed(self):
#         try:
#             self.startAction(self.onShowSortedChanActivationsPressed)
#             epochNum = self.getSelectedEpochNum()
#             imageNum = self.getSelectedImageNum()
#             layerNum = self.blockComboBox.currentIndex() + 1
#             chanNum = self.getSelectedChannelNum()
#             activations = self.netWrapper.getImageActivations(layerNum, imageNum, epochNum)
#             activations = activations[0][chanNum]
#
#             self.clearFigure()
#
#             showCurLayerActivations = True
#             if showCurLayerActivations:
#                 imageData = np.sqrt(activations)
#                 ax = self.figure.add_subplot(self.gridSpec[0, 0])
#                 ax.clear()
#                 ax.imshow(imageData, cmap='plasma')
#
#             vals = attachCoordinates(activations)        # Getting list of (x, y, value)
#             sortedVals = vals[:, vals[2, :].argsort()]
#             curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))
#
#             # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
#             # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
#             allWeights = self.netWrapper.getMultWeights('conv_%d' % layerNum)
#             weights = allWeights[0].numpy()[:, :, :, chanNum]
#             if layerNum == 1:
#                 prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
#             else:
#                 prevLayerActivations = self.netWrapper.getImageActivations(layerNum - 1, imageNum, epochNum)[0]
#                         # Conv2 - 256 * 27 * 27
#
#             if not showCurLayerActivations:
#                 ax = self.figure.add_subplot(self.gridSpec[0, 0])
#                 ax.clear()
#                 ax.hist(weights.flatten(), bins=100)
#                 if ax.get_ylim()[0] > ax.get_ylim()[1]:
#                     ax.invert_yaxis()
#                 ax.set_aspect('auto')
#
#             chanCount = weights.shape[2]
#             showChanCount = int(math.ceil(chanCount / 4))
#             colCount = math.ceil(math.sqrt(showChanCount) * 1.15 / 2) * 2
#
#             # bias = allWeights[1].numpy()[chanNum]
#             chanConvData = conv2D_BeforeSumming(prevLayerActivations, weights)
#             # vals = np.stack([np.arange(chanConvData.shape[0]),
#             #                  np.max(np.abs(chanConvData), axis=(1,2))], axis=0)
#             # prevChansOrder = np.flip(vals[1, :].argsort())
#             vals = np.max(np.abs(chanConvData), axis=(1,2))
#             prevChansOrder = np.flip(vals.argsort())
#             sortedVals = chanConvData[prevChansOrder, :, :]
#             convImageData = layoutLayersToOneImage(sortedVals[:showChanCount], colCount, 1, chanConvData.min())
#             # ax = self.getMainSubplot()
#             ax = self.figure.add_subplot(self.gridSpec[0, 0])
#             ax.clear()
#             im = ax.imshow(convImageData, cmap='plasma')
#             # colorBar = self.figure.colorbar(im, ax=ax)
#
#             options = MultActTops.TMultActOpsOptions()
#             options.layerNum = layerNum
#             actMatrix = self.netWrapper.getImagesActivationMatrix(layerNum)
#             sortedValsList = []
#             for chanInd in range(showChanCount):
#                 sortedImageNums = np.flip(actMatrix[:, prevChansOrder[chanInd]].argsort())[:options.topCount] + 1
#                 sortedValsList.append(sortedImageNums)
#             imageNums = np.stack(sortedValsList, axis=0)
#             self.showTopImages(imageNums, options)
#
#             self.showWeights(self.figure.add_subplot(self.gridSpec[1, 0]), weights[:, :, prevChansOrder],
#                              colCount * 2)
#
#             self.drawFigure()
#         except Exception as ex:
#             self.showProgress("Error: %s" % str(ex))
#
#     def showWeights(self, ax, weights, colCount):
#         weightsImageData = weights.transpose((2, 0, 1))
#         weightsImageData = layoutLayersToOneImage(weightsImageData, colCount, 1, weights.min())
#
#         ax.clear()
#         ax.imshow(weightsImageData, cmap='plasma')
#
#     # Another approach to visualize weights. Doesn't work well, especially with max pool. Because we sum
#     # masks during theirs imposing to generate next layer's mask, but actually the same previous layer's
#     # values are used
#     def showWeightsImage2(self):
#         # try:
#             self.startAction(self.showWeightsImage2)
#             epochNum = self.getSelectedEpochNum()
#             self.netWrapper.loadState(epochNum)
#             layerName = self.getSelectedLayerName()
#
#             imageData = self.netWrapper.calcWeightsVisualization2(layerName)
#             chanCount, colCount = self.getChannelToAnalyzeCount(imageData)
#             imageData = layoutLayersToOneImage(imageData, colCount, 1, imageData.min())
#
#             # dirName = 'Data/WeightsVis/%s_%dChan' % (layerName, chanCount)
#             dirName = 'Data/WeightsVis'
#             if not os.path.exists(dirName):
#                 os.makedirs(dirName)
#             fileName = '%s/Weights_%s_Epoch%d.png' % \
#                        (dirName, layerName, epochNum)
#             imsave(fileName, imageData, format='png')
#             print("Image saved to '%s'" % fileName)
#
#             ax = self.getMainSubplot()
#             ax.clear()
#             ax.imshow(imageData)
#             plt.text(1.1, 0.5, str(epochNum), transform = ax.transAxes)
#             self.drawFigure()
#             return imageData
#         # except Exception as ex:
#         #     self.showProgress("Error: %s" % str(ex))
#
#     def onShowGradientsPressed(self):
#         self.startAction(self.onShowGradientsPressed)
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = self.getSelectedImageNum()
#         layerName = self.getSelectedLayerName()
#
#         data2 = None
#         print("Getting data")
#         if epochNum == self.AllEpochs:
#             dataList = []
#             t0 = datetime.datetime.now()
#             for curEpochNum in self.netWrapper.getSavedNetEpochs():
#                 dataList.append(self.netWrapper.getGradients(layerName, 1, firstImageCount, curEpochNum, True))
#                 t = datetime.datetime.now()
#                 if (t - t0).total_seconds() >= 1:
#                     self.showProgress('Analyzed epoch %d' % curEpochNum)
#                     t0 = t
#                     if self.cancelling or self.exiting:
#                         self.showProgress('Cancelled')
#                         break
#
#             data = np.abs(np.stack(dataList, axis=0))
#             if len(data.shape) == 5:
#                 data2 = np.mean(data, axis=(2))    # Averaging by other convolution channels dimension out of 2
#             data = np.mean(data, axis=(1))
#         else:
#             data = np.abs(self.netWrapper.getGradients(layerName, 1, firstImageCount, epochNum, True))
#
#         self.showProgress('Gradients: %s, max %.4f (%s)' % \
#                 (str(data.shape), data.max(), str([int(v[0]) for v in np.where(data == data.max())])))
#         self.showGradients(data, data2, epochNum == self.AllEpochs)
#
#     def onGradientsByImagesPressed(self):
#         self.startAction(self.onGradientsByImagesPressed)
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = max(self.getSelectedImageNum(), 100)
#         layerName = self.getSelectedLayerName()
#
#         data2 = None
#         dataList = []
#         t0 = datetime.datetime.now()
#         for imageNum in range(1, firstImageCount + 1):
#             dataList.append(self.netWrapper.getGradients(layerName, imageNum, 1, epochNum, True))
#             t = datetime.datetime.now()
#             if (t - t0).total_seconds() >= 1:
#                 self.showProgress('Analyzed image %d' % imageNum)
#                 t0 = t
#                 if self.cancelling or self.exiting:
#                     self.showProgress('Cancelled')
#                     break
#         data = np.abs(np.stack(dataList, axis=0))
#         if len(data.shape) == 5:
#             data2 = np.mean(data, axis=(2))    # Averaging by other convolution channels dimension out of 2
#         data = np.mean(data, axis=(1))
#         # data = np.log10(data)
#         self.showProgress('Gradients: %s, max %.4f (%s)' % \
#                 (str(data.shape), data.max(), str([int(v[0]) for v in np.where(data == data.max())])))
#         self.showGradients(data, data2, True)
#
#     # MultipleObjects means that many images or epochs will be depicted, so a logarithm should be shown
#     def showGradients(self, data, data2, multipleObjects):
#         stdData = None
#         drawMode = 'map'
#         shape = data.shape
#         if len(shape) >= 3:
#             # data = np.reshape(data, (shape[0] * shape[1], shape[2], shape[3]))
#             restAxis = tuple(range(2, len(data.shape)))
#             stdData = np.std(data, axis=restAxis)
#             data = np.mean(data, axis=restAxis)
#         if not data2 is None and len(data2.shape) >= 3:
#             data2 = np.mean(data2, axis=tuple(range(2, len(data2.shape))))
#
#         shape = data.shape
#         if len(shape) == 2:
#             drawMode = '2d'
#             # if shape[1] < 50:
#             #     drawMode = 'plot'
#             #     data = data.flatten()
#             # else:
#             #     data = np.reshape(data, [data.shape[1], 1, 1])
#             #     margin = 0
#         else:
#             data = self.getChannelsToAnalyze(data)
#             margin = self.c_channelMargin
#
#         if multipleObjects and len(data.shape) >= 2:
#             data = data.transpose([1, 0] + list(range(2, len(data.shape))))
#             if not stdData is None:
#                 stdData = stdData.transpose([1, 0] + list(range(2, len(stdData.shape))))
#             if not data2 is None:
#                 data2 = data2.transpose([1, 0] + list(range(2, len(data2.shape))))
#
#         print("Data transformed")
#         self.clearFigure()
#         ax = self.getMainSubplot()
#         ax.clear()
#         if drawMode == 'map':
#             colCount = math.ceil(math.sqrt(data.shape[0]) * 1.15 / 2) * 2
#             data = layoutLayersToOneImage(data,  #np.sqrt(activations),
#                                           colCount, margin, 0)
#             im = ax.imshow(data, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#             ax.title.set_text('Sqrt')
#         elif drawMode == '2d':
#             if multipleObjects:
#                 data = np.abs(data)
#                 data[data < 1e-15] = 1e-15
#                 data = np.log10(data)
#                 ax.title.set_text('Log10')
#             im = ax.imshow(data, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#         else:
#             ax.plot(data)
#
#         if not stdData is None:
#             ax = self.figure.add_subplot(self.gridSpec[0, 0])
#             ax.clear()
#             im = ax.imshow(stdData, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#             ax.title.set_text('Std. dev.')
#
#         ax = self.figure.add_subplot(self.gridSpec[1, 0])
#         ax.clear()
#         if data2 is None:
#             ax.hist(data.flatten(), bins=100)
#             if ax.get_ylim()[0] > ax.get_ylim()[1]:
#                 ax.invert_yaxis()
#             ax.set_aspect('auto')
#         else:
#             if multipleObjects:
#                 data2 = np.abs(data2)
#                 data2[data2 < 1e-15] = 1e-15
#                 data2 = np.log10(data2)
#             im = ax.imshow(data2, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#
#         print("Plots Updated")
#         self.drawFigure()
#         print("Done")
#
#     def onShowWorstImagesPressed(self):
#         try:
#             self.startAction(self.onShowWorstImagesPressed)
#             epochNum = self.getSelectedEpochNum()    # TODO
#             firstImageCount = max(1000, self.getSelectedImageNum())
#             self.netWrapper.loadState(epochNum)     # TODO: to wrap into a NetWrapper's method
#             (losses, predictions) = self.netWrapper._getNet().getImageLosses(1, firstImageCount)
#
#             arr = np.arange(1, firstImageCount + 1)
#             losses = np.stack([arr, losses], axis=0)
#             sortedVals = losses[:, losses[1, :].argsort()]
#
#             imageNums = np.flip(sortedVals[0, -300:])
#             colCount = 20
#             imageData = []
#             for imageNum in imageNums:
#                 curImageData = self.imageDataset.getImage(int(imageNum), 'cropped')
#
#                 if curImageData.max() > 1.01:
#                     if imageNum <= 255:
#                         curImageData[-1][-1] = imageNum
#                 else:
#                     if imageNum <= 999:
#                         curImageData[-1][-1] = imageNum / 1000.0
#
#                 imageData.append(curImageData)
#             imageData = np.stack(imageData, axis=0)
#             imageData = layoutLayersToOneImage(imageData, colCount, 1, imageData.min())
#
#             self.clearFigure()
#
#             ax = self.getMainSubplot()
#             self.showImage(ax, imageData)
#             # im = ax.imshow(imageData, cmap='plasma')
#
#             lossData = np.expand_dims(np.expand_dims(np.flip(sortedVals[1, -300:]), 1), 2)
#             lossData = layoutLayersToOneImage(lossData, colCount, 0, lossData.min())
#
#             ax = self.figure.add_subplot(self.gridSpec[1, 0])
#             ax.clear()
#             im = ax.imshow(lossData, cmap='plasma')
#             colorBar = self.figure.colorbar(im, ax=ax)
#             self.drawFigure()
#
#             self.showProgress("Images: %s" % ', '.join([str(int(num)) for num in imageNums[:10]]))
#             np.savetxt('Data/WorstImageNums.txt', imageNums, fmt='%d', delimiter='')
#             # with open('Data/WorstImageNums.dat', 'wb') as file:
#             #     pickle.dump(imageNums, file)
#         except Exception as ex:
#             self.showProgress("Error: %s" % str(ex))
#
#     def getCorrectlyClassifiedImages(controlObj, classNum, subsetName='train'):
#     #     print(controlObj.netWrapper.batchSize)
#     #     print(list(divideList(controlObj.imageDataset.getClassImageNums(classNum), controlObj.netWrapper.batchSize)))
#         correctImageNums = []
#         classImageNums = controlObj.imageDataset.getClassImageNums(classNum, subsetName=subsetName)
#         for batchImageNums in divideList(classImageNums, controlObj.netWrapper.batchSize):
#             res = controlObj.netWrapper.getImagesActivations_Batch(
#                     None, batchImageNums, epochNum, subsetName=subsetName)
#             predictions = np.argmax(res, axis=1)
#     #         print(predictions)
#             for i in range(len(batchImageNums)):
#                 if predictions[i] == classNum:
#                     correctImageNums.append(batchImageNums[i])
#         print('Correctly classified: %d / %d' % (len(correctImageNums), len(classImageNums)))
#         return correctImageNums
#
#     def getIncorrectlyClassifiedImages(self, classNum, subsetName='train'):
#         resultImageNums = []
#         classImageNums = self.imageDataset.getClassImageNums(classNum, subsetName=subsetName)
#         for batchImageNums in divideList(classImageNums, self.netWrapper.batchSize):
#             res = self.netWrapper.getImagesActivations_Batch(
#                     None, batchImageNums, epochNum, subsetName=subsetName)
#             predictions = np.argmax(res, axis=1)
#     #         print(predictions)
#             for i in range(len(batchImageNums)):
#                 if predictions[i] < 100 or predictions[i] > 300:   # Not dog
#                     resultImageNums.append(batchImageNums[i])
#                     print('%d - %d (%s)' % (batchImageNums[i], \
#                             predictions[i], self.imageDataset.getClassName(predictions[i])))
#         return resultImageNums
#
#
#     def getOcclusionMap(self, imageNum, classNum, occSize=15, subsetName='train'):
#         # try:
#             self.startAction(self.getOcclusionMap)
#             epochNum = self.getSelectedEpochNum()
#             layerName = 'dense_1'    # self.getSelectedLayerName()
#             image = self.imageDataset.getImage(imageNum, self.netWrapper.netPreprocessStageName, subsetName)
#
# #             image2 = image.copy()
# #             activations0 = self.netWrapper.getAuxImageActivations(layerName, image2, epochNum)[0]
# #             activations2 = self.netWrapper.getAuxImageActivations(layerName, image2, epochNum)[0]
# #             diff = activations2 - activations0
# #             print('occ diff 1', diff.min(), diff.max())
# #             print('occ', image.shape)
# #             image[:75, -100:, :] = 100
# #             image[:, :45, :] = image[:, 45:, :]
# #             image[:, -45:, :] = 0
#
# #             image = self.imageDataset.getImage(imageNum, 'cropped', subsetName)
# #             print('im', image.shape, image.min(), image.max())
# #             image = imresize(image[90:, 50:-50, :], (224, 224))
# #             image = np.array(image, dtype=np.float32)
# #             image[:, :, 0] -= 123
# #             image[:, :, 1] -= 116
# #             image[:, :, 2] -= 103
#
#
# #             heatmap = np.zeros(tuple(image.shape[:2]))
#             heatmap = np.zeros((image.shape[0], image.shape[1]))
#             # correct_class = np.argmax(yhat)
#             predClasses = []
#
# #             activations2 = self.netWrapper.getAuxImageActivations(layerName, image2, epochNum)[0]
# #             diff = activations2 - activations0
# #             print('occ diff 2', diff.min(), diff.max())
#
#             for n, (x, y, curImage) in enumerate(self.getOcclusionIter(image, occSize)):
#                 activations = self.netWrapper.getAuxImageActivations(layerName, curImage, epochNum)[0]
#                 heatmap[y : y + occSize, x : x + occSize] = activations[classNum]
#                     # model.predict(curImage.reshape((1, curImage.shape[0], curImage.shape[1], curImage.shape[2])))[0][correct_class]
# #                 print(x,y,n)
#                 predClasses.append(np.argmax(activations))
#                 if n == 0:
#                     debugObjs = [np.copy(image), np.copy(activations)]
# #                     plt.show()
#                 if n < 2:
# #                 if np.argmax(activations) == 160:
# #                 if activations[classNum] > -40:
#                     debugObjs.append(np.copy(curImage))
# #                     ax = plt.imshow(curImage / 64)
# # #                     plt.axes().set_ylabel(str(activations[classNum]), loc='right')
# #                     plt.text(1.02, 0.5, '1, %d: %f' % (n, activations[classNum]), transform=plt.axes().transAxes)
# # #                     ax.legend(str(activations[classNum]), loc='right')
# #                     plt.show()
#
#             print(predClasses)
# #             activations2 = self.netWrapper.getAuxImageActivations(layerName, image2, epochNum)[0]
# #             diff = activations2 - activations0
# #             print('occ diff', diff.min(), diff.max())
#
#             self.debugObjs = debugObjs
# #             class Wrapper(np.ndarray):
# #                 def __init__(self, v, debugImages):
# #                     print('v', v)
# #                     super(Wrapper, self).__init__(v)
# #                     self.debugImages = debugImages
# #                     print(self)
#
# #             return Wrapper(heatmap, debugImages)   # Causes ValueError: maximum supported dimension for an ndarray is 32, found 227
#             return heatmap
#         # except Exception as ex:
#         #     self.showProgress("Error: %s" % str(ex))
#
#     @staticmethod
#     def getOcclusionIter(image, size=8):
#         occlusion = np.full((size * 5, size * 5, 1), [0], np.float32)
#         occlusion_center = np.full((size, size, 1), [0], np.float32)
#         occlusion_padding = size * 2
#
#         image_padded = np.pad(image, tuple([(occlusion_padding, occlusion_padding + size)] * 2 + \
#                                            [(0, 0)] * (len(image.shape) - 2)), \
#                 'constant', constant_values = 0.0)
# #         ax = plt.imshow(image / 64)
# #         plt.text(1.02, 0.5, 'padded %s' % str(image_padded.shape), transform=plt.axes().transAxes)
# #         plt.show()
#
#         for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):
#             for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
#                 tmp = image_padded.copy()
#
# #                 print(x - occlusion_padding, x + occlusion_center.shape[1] + occlusion_padding)
#                 tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
#                     x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
#                     = occlusion
# #                 tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center
#
# #                 if x == occlusion_padding and y == occlusion_padding:
# #                     ax = plt.imshow(tmp / 64)
# #                     plt.text(1.02, 0.5, 'tmp %s' % str(tmp.shape), transform=plt.axes().transAxes)
# #                     plt.show()
# #                     ax = plt.imshow(tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding] / 64)
# #                     plt.text(1.02, 0.5, 'res %s' % str(tmp.shape), transform=plt.axes().transAxes)
# #                     plt.show()
#
#                 yield x - occlusion_padding, y - occlusion_padding, \
#                       tmp[occlusion_padding:image.shape[0] + occlusion_padding, occlusion_padding:image.shape[1] + occlusion_padding]
#
#     @staticmethod
#     def getOcclusionIter2_My_OnlyStarted(image, size=8):
#         occlusion = np.full((size * 5, size * 5, 1), [100], np.float32)
#         occlusion_center = np.full((size, size, 1), [100], np.float32)
#         occlusion_padding = size * 2
#
#         for y in range(0, image.shape[0], size):
#             for x in range(0, image.shape[1], size):
# #                 srcRect = ...
# #                 destRect =
#                 tmp = image_padded.copy()
#
# #                 print(x - occlusion_padding, x + occlusion_center.shape[1] + occlusion_padding)
#                 tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
#                     x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
#                     = occlusion
#
#                 tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center
#                 if x == 0 and y == 0:
#                     ax = plt.imshow(tmp / 64)
#                     plt.text(1.02, 0.5, 'tmp %s' % str(tmp.shape), transform=plt.axes().transAxes)
#                     plt.show()
#                     ax = plt.imshow(tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding] / 64)
#                     plt.text(1.02, 0.5, 'res %s' % str(tmp.shape), transform=plt.axes().transAxes)
#                     plt.show()
#
#
#                 yield x - occlusion_padding, y - occlusion_padding, \
#                       tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
#
#     def doActAnalysis(self, imageNums, layerName,
#                       plotChanCount=100, xShift=0, markerColor='black', chanOrder=None):
#         import torch
#
#         # print(np.where(controlObj.imageDataset.subsets['train'].folders == 'n02109961'))
#         # print(controlObj.imageDataset.getClassNameLabel(752))
#         model = self.netWrapper._getNet(layerName, True)
#         epochNum = self.getSelectedEpochNum()
# #         self.netWrapper.loadState(epochNum)
# #         model.eval()
#         ress = []
#         for imageNum in imageNums:
#         #    img = self.imageDataset.getImage(imageNum, 'net', 'train')
#         #    plt.imshow(img / 255.0 + 0.5)
#             res = controlObj.netWrapper.getImagesActivations_Batch(layerName, [imageNum], epochNum)[0]
#
# #         #     controlObj.netWrapper.pytOptimizer.zero_grad()
# #             with torch.set_grad_enabled(False):
# #             #             print(imgs[:1])
# #             #             res = model(imgs[:1].cuda())
#
# #                 imageData = np.stack([img], axis=0)
# #                 imageData = np.transpose(imageData, (0, 3, 1, 2)) / 128
# #         #         imageData = imageData + np.zeros((1, 3, 227, 227), dtype=np.float32)
# #                 imageData = torch.from_numpy(imageData)
# #                 res = model.forward(imageData.cuda())[0].cpu().numpy()
# # #                 print(res.shape, tuple(range(1, len(res.shape))))
#
#             if len(res.shape) > 1:
#                 res = np.mean(res, axis=tuple(range(1, len(res.shape))))
# #             if correctlyPredicted:    # TODO
#
#             ress.append(res)
#         ress = np.stack(ress, axis=0)
#         if chanOrder is None:
#             order = np.mean(ress, axis=0).argsort()
#         else:
#             order = chanOrder
#         print(order[-10:])
# #         plt.plot(ress[3][order][-plotChanCount:], alpha=0.4)  # One of diagrams just for example
#         for i in range(ress.shape[0]):
#             plt.plot(xShift + np.arange(0, len(ress[i][order][-plotChanCount:])),
#                      ress[i][order][-plotChanCount:], 'o',
#                      color=markerColor,
#                      linewidth=1, alpha=0.03, markersize=1)
# #         plt.twinx().plot(np.std(ress, axis=0)[order][-plotChanCount:], linestyle='--', color='orange', alpha=0.4)
#
#         #     if order is None:
#         #         print(res.shape)
#         #         order = res.argsort()
#         # #         print(order)
#         #         print(order[-10:], res[order][-10:])
#         #     else:
#         #         order2 = res.argsort()
#         # #         print(imageNum, order2[-10:], res[order2][-10:], res[order][-10:])
#         #     plt.plot(res[order][-100:], 'o', linewidth=1, alpha=0.1, markersize=1)
#         #     plt.imshow(res)
# #         plt.show()
#         return order
#
#
#     def onCorrelationAnalyzisPressed(self):
#         self.startAction(self.onCorrelationAnalyzisPressed)
#         self.doCorrelationAnalyzis(False)
#
#     def onCorrelationToOtherModelPressed(self):
#         self.startAction(self.onCorrelationToOtherModelPressed)
#         self.doCorrelationAnalyzis(True)
#
#     def f(self, num=0):
#         fig = figure(figsize=(6, 3))
#         ax = fig.add_subplot(1, 2, 1)
#         ax.plot([5,6,8])
#         ax.legend()
#         ax.title.set_text('test %d' % num)
#         ax = fig.add_subplot(1, 2, 2)
#         # fig2 = figure(figsize=(4, 3))
#         ax.plot([15,6,8])
#
#     def doCorrelationAnalyzis(self, toOtherModel):
#         import CorrelatAnalyzis
#
#         # self.f()
#         matplotlib.rcParams['savefig.dpi'] = 200     # 600 can be too much
#         calculator = CorrelatAnalyzis.CCorrelationsCalculator(self, self.activationCache, self.netWrapper)
#         options = calculator
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = max(1000 if self.netWrapper.name == 'mnist' else 50,
#                               self.getSelectedImageNum())
#         layerName = self.getSelectedLayerName()
#         options.towerCount = self.netWrapper.getTowerCount() * 2
#
#         imagesActs = self.netWrapper.getImagesActivations_Batch(
#                 layerName, range(1, firstImageCount + 1), epochNum)
#         self.showProgress('Activations: %s, max %.4f (%s)' % \
#                 (str(imagesActs.shape), imagesActs.max(),
#                  str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
#         # ests = self.getEstimations(imagesActs)
#
#         # self.f(2)
#         # self.clearFigure()
#         # print('ver 17')
#
#         if 1:
#             # ax = self.figure2.add_subplot()
#             ax = self.getMainSubplot()
#             # ax = figure(figsize=(8, 2)) .add_subplot()
#             varianceDistrs = calculator.getTowersVarianceDistributions(imagesActs)
#             for i in range(varianceDistrs.shape[0]):
#                 ax.plot(varianceDistrs[i], label='%d' % i)
#             ax.legend()
#             ax.title.set_text('Variance distribution for %s' % layerName)
#             self.saveVarianceImage(epochNum, firstImageCount, layerName)
#
#             # self.drawFigure()
#
#         try:
#             # self.drawFigure()
#             if toOtherModel:
#                 imagesActs2 = self.loadOtherModelActivations(firstImageCount, layerName)
#                 calculator.show2ModelsCorrelations(imagesActs, imagesActs2)
#             else:
#                 calculator.showTowersCorrelations(imagesActs)
#         except Exception as ex:
#             self.showProgress("Error on correlations: %s" % str(ex))
#
#         self.drawFigure()
#         self.saveCorrelationsImage(epochNum, firstImageCount, layerName)
#         return varianceDistrs
#
#     def saveCorrelationsImage(self, epochNum, firstImageCount, layerName):
#         dirName = 'Data/Correlations/%s_Correlations_%dImages' % \
#                  (layerName, firstImageCount)
#         if not os.path.exists(dirName):
#             os.makedirs(dirName)
#         fileName = '%s/Corr_%s_Epoch%d.png' % \
#                  (dirName, layerName, epochNum)
#
#         prevDpi = matplotlib.rcParams['savefig.dpi']
#         self.figure.savefig(fileName, format='png', dpi=150)
#         # try:
#             # matplotlib.rcParams['savefig.dpi'] = 600
#
#     def saveVarianceImage(self, epochNum, firstImageCount, layerName):
#         dirName = 'Data/Correlations/%s_Correlations_%dImages' % \
#                  (layerName, firstImageCount)
#         if not os.path.exists(dirName):
#             os.makedirs(dirName)
#         fileName = '%s/Variance_%s_Epoch%d.png' % \
#                  (dirName, layerName, epochNum)
#
#         prevDpi = matplotlib.rcParams['savefig.dpi']
#         self.figure.savefig(fileName, format='png', dpi=150)
#
#     def saveActsForCorrelations(self):
#         epochNum = self.getSelectedEpochNum()
#         firstImageCount = max(2000 if self.netWrapper.name == 'mnist' else 500,
#                               self.getSelectedImageNum())
#         if not os.path.exists('Data/Model2'):
#             os.makedirs('Data/Model2')
#         for layerName in ['conv_2', 'conv_3', 'conv_4', 'dense_1']:
#             imagesActs = self.netWrapper.getImagesActivations_Batch(
#                 layerName, range(1, firstImageCount + 1), epochNum)
#             with open('Data/Model2/Activations_%s.dat' % layerName, 'wb') as file:
#                 pickle.dump(imagesActs, file)
#             self.showProgress('%s activations saved: %s' % \
#                 (layerName, str(imagesActs.shape)))
#
#     def loadOtherModelActivations(self, firstImageCount, layerName):
#         with open('Data/Model2/Activations_%s.dat' % layerName, 'rb') as file:
#             imagesActs = pickle.load(file)
#         return imagesActs[:firstImageCount]
#
#
#     def onChanSpinBoxValueChanged(self):
#         if self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed]:
#             self.lastAction()
#
#     # Main method currently that updates data in accordance with entered image number/channel number
#     # and previously pressed displaying buttons
#     def onSpinBoxValueChanged(self):
#         try:
#             if self.lastAction in [self.onShowImagePressed, self.onShowActivationsPressed, self.onShowActTopsPressed]:
#                 t0 = datetime.datetime.now()
#                 self.onShowActivationsPressed()
#                 self.onShowImagePressed()
#                 # self.showProgress('2 operations: %.1f ms' % ((datetime.datetime.now() - t0).total_seconds() * 1000))
#
#                 # self.lastAction()
#             elif self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed,
#                                      self.onShowGradientsPressed, self.onGradientsByImagesPressed]:
#                 self.lastAction()
#         except Exception as ex:
#             self.showProgress("Error: %s" % str(ex))
#
#     def onEpochComboBoxChanged(self):
#         # if getCpuCoreCount() > 4 and
#         if self.lastAction in [self.onShowMultActTopsPressed, self.onGradientsByImagesPressed,
#                                self.onShowActEstByImagesPressed, self.onShowWorstImagesPressed,
#                                self.onCorrelationAnalyzisPressed, self.onCorrelationToOtherModelPressed]:
#             self.lastAction()
#         else:
#             self.onSpinBoxValueChanged()   # onDisplayPressed()
#
#     def onBlockChanged(self):
#         if self.lastAction in [self.onShowImagePressed,
#                                self.onShowActivationsPressed, self.onShowActTopsPressed]:
#             self.onSpinBoxValueChanged()
#         elif self.lastAction in [self.onShowMultActTopsPressed, self.onShowActEstByImagesPressed,
#                                  self.onShowGradientsPressed, self.onGradientsByImagesPressed,
#                                  self.onCorrelationAnalyzisPressed, self.onCorrelationToOtherModelPressed] and \
#                 self.getSelectedEpochNum() and self.getSelectedEpochNum() > 0:
#             self.lastAction()
#
#
#     class TLearnOptions:
#         def __init__(self, learnRate):
#             self.learnRate = learnRate
#             self.trainImageNums = None
#             self.additTrainImageCount = 0
#
#     def onDoItersPressed(self, iterCount, trainImageNums=None):
#         # epochIterCount = DeepOptions.trainSetSize / DeepOptions.batchSize
#         # c_callIterCount = int(epochIterCount)  # 400
#
#         self.cancelling = False
#         curEpochNum = self.getSelectedEpochNum()
#         if curEpochNum < 0:
#             savedEpochNums = self.netWrapper.getSavedNetEpochs()
#             if savedEpochNums:
#                 curEpochNum = savedEpochNums[-1]
#         print('Cur control object epoch %d, wrapper - %d' % \
#               (self.netWrapper.curEpochNum, curEpochNum))
#         # Here we don\'t reload state' % \
#         if curEpochNum >= 0 and self.netWrapper.curEpochNum < curEpochNum:     # != curEpochNum
#             self.netWrapper.loadState(curEpochNum)
#
#         # if self.weightsReinitEpochNum is None:
#         #     restoreRestEpochCount = 0
#         # else:
#         #     restoreRestEpochCount = 40 - (curEpochNum - self.weightsReinitEpochNum)
#         callback = NetControlObject.TLearningCallback(self, curEpochNum)
#         # callback.learnRate = float(self.learnRateEdit.text())
#         options = NetControlObject.TLearnOptions(self.getLearnRate())
#         if not trainImageNums is None:
#             options.trainImageNums = np.array(trainImageNums, dtype=int)
#             options.additTrainImageCount = max(500, self.getSelectedImageNum() - len(trainImageNums))
#
#         infoStr = self.netWrapper.doLearning(iterCount, options, callback)
#
#         # self.loadNetStateList()
#         # self.epochComboBox.setCurrentIndex(self.epochComboBox.count() - 1)
#         self.showProgress('Epoch %d finished' % self.netWrapper.curEpochNum)
#
#         # restIterCount = iterCount
#         # while restIterCount > 0:
#         #     learnResult = self.netTrader.doLearning(
#         #             c_callIterCount if restIterCount > c_callIterCount else restIterCount)
#         #     restIterCount -= c_callIterCount
#         #     trainIterNum = self.netTrader.getTrainIterNum()
#         #     self.iterNumLabel.setText('Iteration %d' % trainIterNum)
#         #     if self.exiting:
#         #         return
#         #     app.processEvents()
#         #
#         #     # epochNum = trainIterNum / epochIterCount
#         #     # print("Epoch * 3 %%: ", (epochNum * 3) % (DeepOptions.learningRate.circularStepEpochCount * 3))
#         #     # if int(epochNum) % (DeepOptions.learningRate.circularStepEpochCount * 3) == 0:
#         #     #     print("Saving state after %d epochs (%d iterations)" % (epochNum, trainIterNum))
#         #     #     DeepMain.MainWrapper.saveState(self, "States/State_%d_%05d_%f/state.dat" % \
#         #     #             (epochNum, trainIterNum, learnResult))
#
#         # if self.lastAction in [self.onShowMultActTopsPressed]:
#         #     self.lastAction()
#         # else:
#         #     self.onSpinBoxValueChanged()   # onDisplayPressed()
#         self.showProgress(infoStr, False)
#
#     # def onDoItersOnWorstPressed(self, iterCount):
#     def onSpecialDoItersPressed(self, iterCount):
#         if 0:
#             imageNums = np.loadtxt('Data/SelectedWorstImageNums.txt', dtype=int, delimiter='\n')
#             imageNums = list(set(imageNums))
#             print('%d unique worst images numbers read' % len(imageNums))
#             # additImageCount = max(1000, self.imageNumEdit.value()) - len(imageNums)
#             # if additImageCount > 0:
#             #     imageNums = np.concatenate([imageNums, np.random.randint(low=1, high=20000, size=additImageCount)])
#         else:
#             imageNums = np.loadtxt('Data/ImageNums/DistantImageNums1_conv_5.txt', dtype=int, delimiter='\n')
#             imageNums = list(set(imageNums))
#             print('%d unique images numbers read' % len(imageNums))
#
#         self.onDoItersPressed(iterCount, imageNums)
#
#     def getTowerWeights(self):
#         return self.netWrapper.getVariableValue('tower_weights')
#
#     # Expects number like 9990 at channel number spin box (each 0 to 9 - weight from 0 to 100%)
#     def setTowerWeights(self, newWeights):
#         try:
#             # curWeights = self.netWrapper.getMultWeights('tower_weights')
#             curWeights = self.netWrapper.getVariableValue('tower_weights')
#             # intValue = self.chanNumEdit.value()
#             # strValue = '%04d' % intValue
#             # newWeights = [int(ch) / 9.0 for ch in strValue]
#             # newWeights = ([0] * (len(curWeights) - len(newWeights))) + newWeights
#             print('Tower weights: %s, replacing with %s' % (str(curWeights), str(newWeights)))
#             self.netWrapper.setVariableValue('tower_weights', newWeights)
#             self.activationCache.clear()
#         except Exception as ex:
#             self.infoLabel.setText('Error on weights set: %s' % str(ex))
#
#     class TLearningCallback(MnistNetVisWrapper.CBaseLearningCallback):
#         def __init__(self, parent, curEpochNum):
#             super(NetControlObject.TLearningCallback, self).__init__()
#             self.parent = parent
#             self.curEpochNum = curEpochNum
#             # self.restoreRestEpochCount = restoreRestEpochCount
#             self.passedEpochCount = 0
#             self.trainIterNum = -1
#
#         # def __call__(self, infoStr):
#
#         def onBatchEnd(self, trainIterNum, logs):
#             self.logs = logs
#             self.trainIterNum = trainIterNum
#
#             reinitEpochNum = self.parent.weightsReinitEpochNum
#             if reinitEpochNum is not None and reinitEpochNum <= self.curEpochNum and \
#                     self.curEpochNum - reinitEpochNum < 10:
#                 self.parent.restoreReinitedNeirons(True)
#             super(NetControlObject.TLearningCallback, self).onBatchEnd(logs, trainIterNum)
#
#         def onSecondPassed(self):
#             self.logs.pop('batch', None)
#             self.logs.pop('size', None)
#             self.parent.showProgress('Iter %d: %s' %
#                     (self.trainIterNum, str(self.logs)))
#             if self.parent.cancelling or self.parent.exiting:
#                 self.parent.netWrapper.cancelling = True
#
#             reinitEpochNum = self.parent.weightsReinitEpochNum
#             if reinitEpochNum is not None and reinitEpochNum <= self.curEpochNum and \
#                     self.curEpochNum - reinitEpochNum >= 10 and self.curEpochNum - reinitEpochNum < 40:
#                 self.parent.restoreReinitedNeirons()
#             # if self.restoreRestEpochCount > 0:
#             #     self.parent.restoreReinitedNeirons()
#
#         def onEpochEnd(self, curEpochNum, infoStr):
#             self.parent.showProgress(infoStr)
#             # self.restoreRestEpochCount -= 1
#             self.curEpochNum = curEpochNum
#             self.passedEpochCount += 1
#
#             self.lastUpdateTime = datetime.datetime.now() + datetime.timedelta(seconds=2)
#                 # A hacking of super-class in order to make epoch info visible longer
#
#     # Tries several learning rate and selects the one that gave biggest advance
#     def tryDiffParams(self, learnRates):  # tryCount, minLearnRate, maxLearnRate):
#         startEpochNum = self.netWrapper.curEpochNum
#         bestResult = None
#         tryCount = len(learnRates)
#         for tryInd in range(tryCount):
#             print('try ', tryInd)
#             if tryInd > 0:
#                 self.loadState(startEpochNum)
#
#             curLearnRate = learnRates[tryInd]
#             # if tryCount > 1:
#             #     curLearnRate = minLearnRate + (maxLearnRate - minLearnRate) / (tryCount - 1)
#             # else:
#             #     curLearnRate = minLearnRate
#
#             self.learnRate = curLearnRate
#             self.curEpochNum = startEpochNum
#             self.onDoItersPressed(200)
#             if bestResult is None or bestResult > self.netWrapper.lastEpochResult['TrainLoss']:
#                 bestResult = self.netWrapper.lastEpochResult['TrainLoss']
#                 bestParam = curLearnRate
#                 self.saveState()
#
#         self.netWrapper.printProgress("Best try's learn rate %f" % bestParam)
#         self.loadState(startEpochNum + 1)
#
#     def onTestPress(self):
#         # self.loadState()
#         pass
#
#     def onLoadStatePressed(self):
#         self.loadState(self.getSelectedEpochNum())
#
#     def onReinitWorstNeironsPressed(self):
#         self.startAction(self.onReinitWorstNeironsPressed)
#         curEpochNum = self.getSelectedEpochNum()
#         if curEpochNum < 0:
#             curEpochNum = self.netWrapper.getSavedNetEpochs()[-1]
#         self.weightsBeforeReinit = dict()
#         self.weightsAfterReinit = dict()
#         self.weightsReinitInds = dict()
#
#         if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper):
#             self.reinitWorstNeirons_Random(curEpochNum)
#         else:  # if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper3_Towers):
#             # self.reinitWorstNeirons_OneLayerTowers(curEpochNum)
#             self.reinitWorstNeirons_SeparatedTowers(curEpochNum)
#
#         self.weightsReinitEpochNum = curEpochNum
#         self.infoLabel.setText('Worst neirons reinitialized')
#
#     def reinitWorstNeirons_Random(self, curEpochNum):
#         for layerName in self.netWrapper.getComponentNetLayers():
#             gradients = self.netWrapper.getGradients(layerName, 1, 500, curEpochNum)
#             weights = self.netWrapper.getMultWeights(layerName)
#
#             std0 = np.std(weights)
#             gradients2 = np.square(gradients)
#             resetShape = list(gradients.shape)
#             self.weightsBeforeReinit[layerName] = weights
#             if layerName[:4] == 'conv':
#                 meanGradients = np.mean(gradients2, axis=(0, 2, 3))
#                 sortedInds = meanGradients.argsort()
#                 indsToChange = sortedInds[4 : ]
#                 others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)], :, :]
#                 othersStdDev = np.std(others)
#                 resetShape[1] = len(indsToChange)
#                 weights[:, indsToChange, :, :] = \
#                     np.random.normal(scale=othersStdDev / 16, size=resetShape)
#             elif layerName[:5] == 'dense':
#                 meanGradients = np.mean(gradients2, axis=(0, ))
#                 sortedInds = meanGradients.argsort()
#                 if 0:
#                     indsToChange = sortedInds[12 if layerName == 'dense_1' else 1 : ]
#                     others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)]]
#                 else:
#                     indsToChange = sortedInds
#                     others = weights
#                 othersStdDev = np.std(others)
#                 resetShape[1] = len(indsToChange)
#                 weights[:, indsToChange] = \
#                     np.random.normal(scale=othersStdDev / 16, size=resetShape)
#             else:
#                 indsToChange = None
#                 del self.weightsBeforeReinit[layerName]
#
#             if not indsToChange is None:
#                 self.netWrapper.setMultWeights(layerName, weights)
#                 self.weightsReinitInds[layerName] = indsToChange
#                 self.weightsAfterReinit[layerName] = weights
#                 print("%s reinit: std. %.9f - %.9f, neirons %s" % \
#                       (layerName, std0, np.std(weights),
#                        ', '.join(sorted([str(i) for i in indsToChange]))))
#
#     def reinitWorstNeirons_OneLayerTowers(self, curEpochNum):
#         towerCount = self.netWrapper.getMultWeights('conv_1').shape[1]          # E.g., 80, weights - 1 * 80 * 5 * 5
#         gradients = self.netWrapper.getGradients('conv_3', 1, 500, curEpochNum)    # E.g. (80 * 4) * 2 * 3 * 3
#         gradients2 = np.reshape(np.square(gradients), (towerCount, -1, gradients.shape[2], gradients.shape[3]))
#         meanGradients = np.mean(gradients2, axis=(1, 2, 3))
#         sortedInds = meanGradients.argsort()
#         towerIndsToChange = sortedInds[8 : ]
#         for layerName in self.netWrapper.getNetLayersToVisualize():
#             gradients = self.netWrapper.getGradients(layerName, 1, 500, curEpochNum)
#             weights = self.netWrapper.getMultWeights(layerName)
#
#             std0 = np.std(weights)
#             resetShape = list(gradients.shape)
#             self.weightsBeforeReinit[layerName] = weights
#             if layerName == 'conv_1':
#                 indsToChange = towerIndsToChange
#                 others = weights[:, sortedInds[: gradients.shape[1] - len(indsToChange)], :, :]
#                 othersStdDev = np.std(others)
#                 resetShape[1] = len(indsToChange)
#                 weights[:, indsToChange, :, :] = \
#                     np.random.normal(scale=othersStdDev, size=resetShape)
#             elif layerName[:4] == 'conv':
#                 towerWidth = weights.shape[0] // towerCount
#                 indsToChange = []
#                 for i in range(towerWidth):
#                     indsToChange.append(towerIndsToChange * towerWidth + i)
#                 indsToChange = np.concatenate(indsToChange, axis=0)
#
#                 others = weights[sortedInds[ : gradients.shape[0] - len(indsToChange)], :, :, :]
#                 othersStdDev = np.std(others)
#                 resetShape[0] = len(indsToChange)
#                 weights[indsToChange, :, :, :] = \
#                     np.random.normal(scale=othersStdDev, size=resetShape)
#             elif layerName[:5] == 'dense':
#                 gradients2 = np.square(gradients)
#                 meanGradients = np.mean(gradients2, axis=(0, ))
#                 sortedInds = meanGradients.argsort()
#                 if 0:
#                     indsToChange = sortedInds[12 if layerName == 'dense_1' else 1 : ]
#                     others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)]]
#                 else:
#                     indsToChange = sortedInds
#                     others = weights
#                 othersStdDev = np.std(others)
#                 resetShape[1] = len(indsToChange)
#                 weights[:, indsToChange] = \
#                     np.random.normal(scale=othersStdDev, size=resetShape)
#             else:
#                 indsToChange = None
#                 del self.weightsBeforeReinit[layerName]
#
#             if not indsToChange is None:
#                 self.netWrapper.setMultWeights(layerName, weights)
#                 self.weightsReinitInds[layerName] = indsToChange
#                 self.weightsAfterReinit[layerName] = weights
#                 print("%s reinit: std. %.9f - %.9f, neirons %s" % \
#                       (layerName, std0, np.std(weights),
#                        ', '.join(sorted([str(i) for i in indsToChange]))))
#
#     # For CMnistModel3_Towers
#     def reinitWorstNeirons_SeparatedTowers(self, curEpochNum):
#         towerInd = 0
#         towerMeans = []
#         while True:
#             try:
#                 gradients = self.netWrapper.getGradients('conv_2_%d' % towerInd, 1, 500, curEpochNum)
#             except Exception as ex:
#                 break   # No more towers
#
#             gradients2 = np.square(gradients)         # E.g. 40 * 24 * 3 * 3
#             meanGradients = np.mean(gradients2) # , axis=(1, 2, 3))
#             towerMeans.append(meanGradients)
#             towerInd += 1
#         if towerInd <= 0:
#             raise Exception('No towers found')
#
#         towerMeans = np.stack(towerMeans)
#         sortedTowerInds = towerMeans.argsort()
#         towerIndsToChange = sortedTowerInds[4:]
#         towersToPreserveInds = sortedTowerInds[:4]
#         for towerInd in towerIndsToChange:
#             for layerName in ['conv_1_%d' % towerInd, 'conv_2_%d' % towerInd]:
#                 weights = self.netWrapper.getMultWeights(layerName)
#                 std0 = np.std(weights)
#                 resetShape = list(weights.shape)
#                 # self.weightsBeforeReinit[layerName] = weights
#
#                 others = weights # [:, sortedInds[: gradients.shape[1] - len(indsToChange)], :, :]
#                 othersStdDev = np.std(others)
#                 weights[:, :, :, :] = \
#                     np.random.normal(scale=othersStdDev / 16, size=resetShape)
#                 self.netWrapper.setMultWeights(layerName, weights)
#
#         if 1:
#             layerName = 'conv_3'
#             # gradients = self.netWrapper.getGradients(layerName, 500, curEpochNum)
#             weights = self.netWrapper.getMultWeights(layerName)
#
#             std0 = np.std(weights)
#             # gradients2 = np.square(gradients)
#             resetShape = list(weights.shape)
#
#             indsToChange = []
#             assert weights.shape[0] % len(sortedTowerInds) == 0
#             towerWidth = weights.shape[0] // len(sortedTowerInds)
#             for towerInd in towerIndsToChange:
#                 indsToChange += range(towerInd * towerWidth, (towerInd + 1) * towerWidth)
#             # meanGradients = np.mean(gradients2, axis=(0, 2, 3))
#             # sortedInds = meanGradients.argsort()
#             resetShape[0] = len(indsToChange)
#             indsToPreserve = set(range(len(indsToChange)))
#             indsToPreserve = list(indsToPreserve.difference(indsToChange))
#
#             # self.weightsBeforeReinit[layerName] = weights
#             weights[indsToChange, :, :, :] = \
#                 np.random.normal(scale=othersStdDev / 16, size=resetShape)
#             weights[indsToPreserve, :, :, :] *= 4
#             # self.weightsReinitInds[layerName] = indsToChange
#             # self.weightsAfterReinit[layerName] = weights
#
#         keepLayerNames = ['dense_1', 'dense_2']
#         for towerInd in towersToPreserveInds:
#             for layerName in ['conv_1_%d' % towerInd, 'conv_2_%d' % towerInd]:
#                 keepLayerNames.append(layerName)
#         for layerName in keepLayerNames:
#             weights = self.netWrapper.getMultWeights(layerName)
#             self.weightsBeforeReinit[layerName] = weights
#             self.weightsReinitInds[layerName] = []
#             self.weightsAfterReinit[layerName] = weights
#             # print("%s reinit: std. %.9f - %.9f, neirons %s" % \
#             #       (layerName, std0, np.std(weights),
#             #        ', '.join(sorted([str(i) for i in indsToChange]))))
#
#     def restoreReinitedNeirons(self, veryStrong=False):
#         # print('restoreReinitedNeirons %s' % ('very strong' if veryStrong else ''))
#         if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper):
#             indicesAxis = 1
#         else:
#             indicesAxis = 0
#
#         for layerName in self.netWrapper.getComponentNetLayers():
#             if layerName in self.weightsBeforeReinit:
#                 # if veryStrong and layerName[:4] == 'conv':
#                 #     self.netWrapper.setMultWeights(layerName, self.weightsAfterReinit[layerName])
#                 #     continue
#
#                 weights = self.netWrapper.getMultWeights(layerName)
#                 indsToChange = self.weightsReinitInds[layerName]
#                 indsToPreserve = set(range(len(indsToChange)))
#                 indsToPreserve = list(indsToPreserve.difference(indsToChange))
#                 print("Restoring %d weights at %s" % (len(indsToPreserve), layerName))
#
#                 if layerName[:4] == 'conv':
#                     # if layerName == 'conv_3':
#                     #     w
#                     if indicesAxis == 1 or layerName == 'conv_1':    # Restoring after reinitWorstNeirons_Random
#                                                                      # or conv_1 after reinitWorstNeirons_OneLayerTowers
#                         weights[:, indsToPreserve, :, :] = \
#                             self.weightsBeforeReinit[layerName][:, indsToPreserve, :, :]
#                     else:                                            # Restoring after reinitWorstNeirons_OneLayerTowers
#                         weights[indsToPreserve, :, :, :] = \
#                             self.weightsBeforeReinit[layerName][indsToPreserve, :, :, :]
#                     self.netWrapper.setMultWeights(layerName, weights)
#                 elif layerName[:5] == 'dense':
#                     weights[:, indsToPreserve] = \
#                         self.weightsBeforeReinit[layerName][:, indsToPreserve]
#                     self.netWrapper.setMultWeights(layerName, weights)
#
#
#     def saveState(self):
#         try:
#             self.netWrapper.saveState()
#         except Exception as ex:
#             self.showProgress("Error in saveState: %s" % str(ex))
#
#     def loadNetStateList(self):
#         try:
#             self.savedNetEpochs = self.netWrapper.getSavedNetEpochs()
#         except Exception as ex:
#             self.showProgress("Error in loadState: %s" % str(ex))
#             self.savedNetEpochs = []
#         self.savedNetEpochs = ['---  All  ---'] + self.savedNetEpochs
#
#     def loadState(self, epochNum=None):
#         try:
#             if epochNum is None:
#                 epochNum = -1      # When there is only one file and its epoch is unknown
#             self.netWrapper.loadState(epochNum)
#             # self.netWrapper.loadCacheState()
#             self.showProgress('Epoch %d' % self.netWrapper.curEpochNum)
#         except Exception as ex:
#             self.showProgress("Error in loadState: %s" % str(ex))
#
#     def loadCacheState(self):
#         try:
#             self.netWrapper.loadCacheState()
#         except Exception as ex:
#             self.showProgress("Error in loadState: %s" % str(ex))
#
#
#     def mouseMoveEvent(self, event):
#         try:
#             s = 'Mouse move: %d, %d' % (int(event.x), int(event.y))
#             self.infoLabel.setText(s)
#             return
#
#             editBlockInd = None
#             if event.inaxes and event.inaxes.get_navigate():
#                 try:
#                     s = event.inaxes.format_coord(event.xdata, event.ydata)
#                     if event.inaxes in self.curDiags:
#                         editBlockInd = self.curDiags[event.inaxes]
#                     else:
#                         editBlockInd = -1
#                     s = 'Block %d, %s' % (editBlockInd, s)
#                 except:
#                     pass
#
#                 else:
#                     artists = [a for a in event.inaxes.mouseover_set
#                                if a.contains(event) and a.get_visible()]
#                     if artists:
#                         a = max(artists, key=lambda x: x.zorder)
#                         if a is not event.inaxes.patch:
#                             data = a.get_cursor_data(event)
#                             if data is not None:
#                                 s += ' [%s]' % a.format_cursor_data(data)
#
#                     # if len(self.mode):
#                     #     s = '%s, %s' % (self.mode, s)
#                     # else:
#             self.infoLabel.setText('Mouse move: %s' % s)
#
#             if event.button == 1:
#                 self.onDiagMouseChange(editBlockInd, event.xdata, event.ydata, event.y)
#         except Exception as ex:
#             # print("Exception in mouseMove: ", str(ex))
#             pass  # Sometimes we are getting MouseMoveEvent object without x and y
#
#     def mousePressEvent(self, event):
#         return
#         if event.inaxes and event.inaxes.get_navigate():
#             try:
#                 x = int(event.xdata)
#                 y = int(event.ydata)
#                 if event.inaxes in self.curDiags:
#                     editBlockInd = self.curDiags[event.inaxes]
#                 else:
#                     editBlockInd = -1
#                 self.mousePressPos = (editBlockInd, event.xdata, event.ydata, event.y)
#             except:
#                 pass
#         else:
#             self.infoLabel.setText('Mouse press: no data selected')
#
#     def onDiagMouseChange(self, editBlockInd, x, y, screenY):
#         if self.mousePressPos is None:
#             self.mousePressPos = (editBlockInd, x, y, screenY, self.getClosestY(editBlockInd, x, y))
#             return
#         editBlockInd = self.mousePressPos[0]
#         x = int(self.mousePressPos[1] + 0.5)
#         editElemY = self.mousePressPos[4]
#         sampleBlock = self.curEditedBlocks[editBlockInd + 1]
#         blockShape = sampleBlock.shape
#         print("blockShape %s, start %s, x %d, screen y %d, selected y %d" %
#               (str(blockShape), str(self.mousePressPos), x, screenY, editElemY))
#         if x < 0:
#             x = 0
#         if x >= blockShape[1]:
#             x = blockShape[1] - 1
#         if editElemY < 0:
#             editElemY = 0
#         if editElemY >= blockShape[0]:
#             editElemY = blockShape[0] - 1
#
#         self.curEditedBlocks[editBlockInd + 1][editElemY][x] = \
#                 self.curInitialBlocks[editBlockInd + 1][editElemY][x] + \
#                 (screenY - self.mousePressPos[3]) / 300
#
#         print("V ", self.curEditedBlocks[editBlockInd + 1][editElemY][x])
#
#         # self.clearFigure()
#         ax = self.figure.add_subplot(331)
#         ax.clear()
#         ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[0], color=(0.3, 0.3, 0.8, 0.5), linestyle = 'dashed')
#         ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[1], color=(0.7, 0.3, 0.3, 0.5))
#         ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[2], color=(0.3, 0.7, 0.3, 0.5))
#         self.drawFigure()
#
#     def getClosestY(self, editBlockInd, x, y):
#         intX = int(x + 0.5)
#         sampleBlock = self.curEditedBlocks[editBlockInd + 1]
#         minDist = None
#         minDistElemY = None
#         for elemY in range(sampleBlock.shape[0]):
#             dist = abs(sampleBlock[elemY][intX] - y)
#             print("Dist %d: %f - %f" % (elemY, sampleBlock[elemY][intX], y))
#             if elemY == 0 or minDist > dist:
#                 minDist = dist
#                 minDistElemY = elemY
#         return minDistElemY
#
#     def mouseReleaseEvent(self, event):
#         if self.mousePressPos is None:
#             return
#
#         self.mousePressPos = None
#         return
#
#         samples = self.getSamples()
#         if not samples:
#             return
#         sample = (np.expand_dims(self.curEditedBlocks[0].transpose(), 0),
#                   samples[1][:1, :], samples[2][:1, :])
#         feedDict = self.netTrader.getNetFeedDict_SamplesRange('train', sample)
#
#         newResults = [sample]
#         for intermBlockInd in range(self.net.getIntermResultsBlockCount()):
#             block = self.net.getIntermResultsBlock(intermBlockInd, feedDict)
#             if self.prevDrawnResults is not None:
#                 self.drawSamplesIntermResults(self.prevDrawnResults[intermBlockInd + 1],
#                                               intermBlockInd, 332 + intermBlockInd, True)
#             self.drawSamplesIntermResults(block, intermBlockInd, 332 + intermBlockInd,
#                                           False, self.prevDrawnResults is None)
#                     # int(self.plotNumLineEdit.text()))
#             self.curInitialBlocks.append(block)
#             newResults.append(block)
#         self.prevDrawnResults = newResults
#         self.infoLabel.setText('Results updated')
#
#     def plotSamplesResults(self, testSamplesActuals, testSamplesPreds):
#         import random
#
#         if 0:
#             ''' plot some random stuff '''
#
#             self._static_ax = self.canvas.figure.subplots()
#             t = np.linspace(0, 10, 501)
#             self._static_ax.plot(t, np.tan(t), ".")
#
#
#         # data = [random.random() for i in range(10)]
#
#         (testSamplesActuals, testSamplesPreds) = self.sortByClass(testSamplesActuals, testSamplesPreds)
#         (_, testActualClassNums) = np.where(testSamplesActuals==1)
#
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         # ax.plot(data, '*-')
#         ax.matshow(np.matrix.transpose(testSamplesPreds), aspect='auto', cmap="YlGn")
#         # ax.matshow(np.matrix.transpose(testSamplesActuals), aspect='auto', \
#         #            cmap="YlGn", hatch='/', alpha=0.5)
#         del self.curDiags[ax.axes]
#         ax.scatter(np.arange(testSamplesActuals.shape[0]), testActualClassNums, alpha=0.5)
#         self.drawFigure()
#
#     def drawSamplesResults(self, datasetName):
#         if self.net:
#             samples = self.getSamples()
#             # feedDict = self.netTrader.getNetFeedDict(datasetName.lower(), 300)
#             feedDict = self.netTrader.getNetFeedDict_SamplesRange(datasetName, samples)
#             (samplesActuals, samplesPreds) = self.net.getSamplesResults(feedDict)
#             for otherNet in self.otherNets:
#                 (otherActuals, otherPreds) = otherNet.getSamplesResults(feedDict)
#                 if not np.array_equal(samplesActuals, otherActuals):
#                     print("samplesActuals != otherActuals")
#                 samplesPreds = np.concatenate((samplesPreds, otherPreds), axis=1)
#         else:  # Fast init
#             (samplesActuals, samplesPreds) = self.createSamplesResultsTestData()
#
#         self.plotSamplesResults(samplesActuals, samplesPreds)
#         self.infoLabel.setText('Displayed data for dataset %s' % datasetName)
#
#     def drawGradients(self, datasetName, gradBlockInd, plotNum = 111, \
#                                  startSampleInd = None, endSampleInd = None):
#         samples = self.getSamples(startSampleInd, endSampleInd)
#         # feedDict = self.netTrader.getNetFeedDict(datasetName.lower(), 300)
#         feedDict = self.netTrader.getNetFeedDict_SamplesRange(datasetName, samples)
#         block = self.net.getGradientsBlock(gradBlockInd, feedDict)
#         self.infoLabel.setText('Block size: %s' % str(block.shape))
#         block = np.squeeze(block)
#
#         cdict = {
#           'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
#           'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
#           'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
#         }
#
#         if 0:
#             cm = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
#
#             x = np.arange(0, 10, .1)
#             y = np.arange(0, 10, .1)
#             X, Y = np.meshgrid(x,y)
#
#             data = 2*( np.sin(X) + np.sin(3*Y) )
#             f =  lambda x:np.clip(x, -4, 0)
#
#         ax = self.figure.add_subplot(plotNum)
#         ax.clear()
#         # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)
#         # ax.plot(data, '*-')
#         while len(block.shape) > 2:
#             block = np.squeeze(block[0])
#         self.im = ax.matshow(np.matrix.transpose(block), aspect='auto', cmap="YlGn")
#         self.curDiags[ax.axes] = gradBlockInd      # TODO? to replace with del self.curDiags[ax.axes]
#         # ax.matshow(np.matrix.transpose(testSamplesActuals), aspect='auto', \
#         #            cmap="YlGn", hatch='/', alpha=0.5)
#         # scatter = plt.scatter([1, 2], [30, 4], cmap="YlGn")
#         # scatter.set_array(np.array([block.min(), block.max()]))
#         # self.figure.colorbar(scatter)
#         # plt.colorbar()
#
#         print("Stddev %f, mean %f" % (np.std(block), np.mean(block)))
#         if np.std(block) > 0.0015:
#             mult = 1000
#         else:
#             mult = 10000
#
#         absMeanByInput = np.mean(abs(block), axis=1)
#         absMeanByOutput = np.mean(abs(block), axis=0)
#         # font = {'fontname':'Arial', 'size':'7'}
#         for i in range(block.shape[0]):
#             for j in range(block.shape[1]):
#                 multV = block[i, j] * mult
#                 if abs(multV) >= 2:
#                     text = ax.text(i, j, int(multV),
#                                    ha="center", va="center", color="w", fontsize=7)
#             ax.text(i, block.shape[1] + 0.5, int(absMeanByInput[i] * 1e5),
#                     ha="center", va="center", color="b", fontsize=7)
#             # print("Mean abs: ", np.mean(abs(blockGradients), axis=1))
#         for j in range(block.shape[1]):
#             ax.text(block.shape[0] + 0.5, j, int(absMeanByOutput[j] * 1e5),
#                     ha="center", va="center", color="b", fontsize=7)
#
#         if gradBlockInd == 0:
#             (samplesActuals, _) = self.net.getSamplesResults(feedDict)
#             (_, actualClassNums) = np.where(samplesActuals==1)
#             ax.scatter(np.arange(actualClassNums.shape[0]), actualClassNums, alpha=0.5)
#
#         self.drawFigure()
#
#     def drawSamplesIntermResults(self, block, intermBlockInd, plotNum = 111,
#                                  arePrevResults = False, clearExisting = True):
#         self.infoLabel.setText('Block size: %s' % str(block.shape))
#
#         ax = self.figure.add_subplot(plotNum)
#         if clearExisting:
#             ax.clear()
#         block = np.squeeze(block)
#         if len(block.shape) == 1:
#             ax.plot(np.arange(block.shape[0]), block, '*-', \
#                     color=((0, 0.3, 0.6, 0.4) if arePrevResults else (0, 0, 0.8, 0.7)), \
#                     linestyle=(':' if arePrevResults else '-'))
#         else:
#             # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)
#             # ax.plot(data, '*-')
#             if arePrevResults:
#                 return
#             self.im = ax.matshow(np.matrix.transpose(block), aspect='auto', cmap="YlGn")
#         self.curDiags[ax.axes] = intermBlockInd
#
#         print("Stddev %f, mean %f" % (np.std(block), np.mean(block)))
#         if 0:
#             if np.std(block) > 0.0015:
#                 mult = 1000
#             else:
#                 mult = 10000
#
#             absMeanByInput = np.mean(abs(block), axis=1)
#             absMeanByOutput = np.mean(abs(block), axis=0)
#             # font = {'fontname':'Arial', 'size':'7'}
#             for i in range(block.shape[0]):
#                 for j in range(block.shape[1]):
#                     multV = block[i, j] * mult
#                     if abs(multV) >= 2:
#                         text = ax.text(i, j, int(multV),
#                                        ha="center", va="center", color="w", fontsize=7)
#                 ax.text(i, block.shape[1] + 0.5, int(absMeanByInput[i] * 1e5),
#                         ha="center", va="center", color="b", fontsize=7)
#                 # print("Mean abs: ", np.mean(abs(blockGradients), axis=1))
#             for j in range(block.shape[1]):
#                 ax.text(block.shape[0] + 0.5, j, int(absMeanByOutput[j] * 1e5),
#                         ha="center", va="center", color="b", fontsize=7)
#
#         self.drawFigure()
#
#     def onDisplayPressed(self):
#         # self.onDatasetChanged(self.datasetComboBox.currentIndex())
#         datasetName = self.datasetComboBox.itemText(self.datasetComboBox.currentIndex())
#         try:
#             blockInd = self.blockComboBox.currentIndex()
#             blockName = self.blockComboBox.itemText(blockInd)
#             # result = re.search('Gradient.* (\d+)\. ', blockName)
#             # if result:
#             #     self.drawGradients(datasetName, int(result.group(1)), int(self.plotNumLineEdit.text()))
#             # elif blockName == "Samples results":
#             #     self.drawSamplesResults(datasetName)
#         except Exception as ex:
#             print('Exception in onDisplayPressed: %s' % str(ex))
#
#     def onDisplayIntermResultsPressed(self):
#         samples = self.getSamples()
#         if not samples:
#             return
#         sample = (samples[0][:1, :, :], samples[1][:1, :], samples[2][:1, :])
#         feedDict = self.netTrader.getNetFeedDict_SamplesRange('train', sample)
#
#         # intermBlockInd = self.blockComboBox.currentIndex()
#         # if hasattr(sample, 'realDiffs'):
#         #         plt.plot(sample.realDiffs, color=(0, 0.8, 0))
#         self.clearFigure()
#         ax = self.figure.add_subplot(331)
#         ax.clear()
#         # if self.prevDrawnResults is not None:
#         #     ax.plot(np.arange(sample[1].shape[1]), self.prevDrawnResults[0][1][0, :], color=(0.3, 0.3, 0.8, 0.3), linestyle = ':')
#         #     ax.plot(np.arange(sample[0].shape[1]), self.prevDrawnResults[0][0][0, :, 1], color=(0.7, 0.3, 0.3, 0.3), linestyle = ':')
#         #     ax.plot(np.arange(sample[0].shape[1]), self.prevDrawnResults[0][0][0, :, 2], color=(0.3, 0.7, 0.3, 0.3), linestyle = ':')
#         ax.plot(np.arange(sample[1].shape[1]), sample[1][0, :], color=(0.3, 0.3, 0.8, 0.65), linestyle = 'dashed')
#         ax.plot(np.arange(sample[0].shape[1]), sample[0][0, :, 1], color=(0.7, 0.3, 0.3, 0.65))
#         ax.plot(np.arange(sample[0].shape[1]), sample[0][0, :, 2], color=(0.3, 0.7, 0.3, 0.65))
#         print("Sample %s %s" % (str(sample[0].shape), str(sample[1].shape)))
#         self.curInitialBlocks = [sample[0][0].transpose()] #np.stack([sample[1][0, :], sample[0][0, :, 1], sample[0][0, :, 2]])]
#
#         newResults = [sample]
#         #d_ for intermBlockInd in range(self.net.getIntermResultsBlockCount()):
#         #     block = self.net.getIntermResultsBlock(intermBlockInd, feedDict)
#         #     self.drawSamplesIntermResults(block, intermBlockInd, 332 + intermBlockInd)
#         #             # int(self.plotNumLineEdit.text()))
#         #     self.curInitialBlocks.append(block)
#         #     newResults.append(block)
#         self.curEditedBlocks = [copy.deepcopy(arr) for arr in self.curInitialBlocks]
#         self.prevDrawnResults = newResults
#
#         sampleInd = int(self.startSampleIndLineEdit.text())
#         self.startSampleIndLineEdit.setText(str(int(sampleInd + 1)))
#
#     def onDatasetChanged(self, datasetInd):
#         self.onDisplayPressed()
#
#         # datasetName = self.datasetComboBox.itemText(datasetInd)
#         # self.drawSamplesResults(datasetName)
#
#     def onIncreaseLearningRatePressed(self):
#         self.net.changeRate(2)
#         self.infoLabel.setText('Learning rate increased')
#
#     def getSamples(self, startSampleInd = None, endSampleInd = None):
#         datasetName = self.datasetComboBox.itemText(self.datasetComboBox.currentIndex())
#         if self.curSavedBatchDatasetName != datasetName:
#             dataset = self.netTrader.getDatasetByName(datasetName.lower())
#             sampleToSaveCount = 10000
#             if sampleToSaveCount > len(dataset):
#                 sampleToSaveCount = len(dataset)
#             self.curSavedBatch = dataset.getBatch(sampleToSaveCount)
#             print("Loaded %d samples from %s" % (len(self.curSavedBatch[0]), datasetName))
#             self.curSavedBatchDatasetName = datasetName
#         if startSampleInd is None:
#             startSampleInd = int(self.startSampleIndLineEdit.text())
#         if endSampleInd is None:
#             endSampleInd = int(self.endSampleIndLineEdit.text())
#         return tuple([arr[startSampleInd : endSampleInd] for arr in self.curSavedBatch])
#
#     def loadData(self):
#         # with open('Block.dat', "rb") as file:
#         #     self.block = pickle.load(file)
#         pass
#
#
#
# def imshow_grid(data, height=None, width=None, normalize=False, padsize=1, padval=0):
#     '''
#     Take an array of shape (N, H, W) or (N, H, W, C)
#     and visualize each (H, W) image in a grid style (height x width).
#     '''
#     if normalize:
#         data -= data.min()
#         data /= data.max()
#
#     N = data.shape[0]
#     if height is None:
#         if width is None:
#             height = int(np.ceil(np.sqrt(N)))
#         else:
#             height = int(np.ceil( N / float(width) ))
#
#     if width is None:
#         width = int(np.ceil( N / float(height) ))
#
#     assert height * width >= N
#
#     # append padding
#     padding = ((0, (width*height) - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
#     data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
#
#     # tile the filters into an image
#     data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#     data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])
#
#     plt.imshow(data)
#
#
# controlObj = NetControlObject()
# # controlObj.show()
#
# controlObj.init()
# #controlObj.loadCacheState()
#
# # controlObj.loadState(controlObj.getSelectedEpochNum())
# # controlObj.netWrapper.getGradients()
# # MnistNetVisWrapper.getGradients(controlObj.netWrapper.net.model)
# #controlObj.onDoItersPressed(1)
# # controlObj.onReinitWorstNeironsPressed()
#
# # controlObj.onShowImagePressed()
# # controlObj.onDisplayIntermResultsPressed()
# # controlObj.onDisplayPressed()
# # controlObj.onShowActivationsPressed()
# # controlObj.onShowMultActTopsPressed()
# # controlObj.onShowSortedChanActivationsPressed()
# # controlObj.onSpinBoxValueChanged()
# # controlObj.buildMultActTops_MultiThreaded()
# # controlObj.onGradientsByImagesPressed()
# # controlObj.onShowWorstImagesPressed()
# # controlObj.onShowImagesWithTSnePressed()
# # controlObj.onCorrelationToOtherModelPressed()
#
#
# epochNum = -3
# imageNum = 5
# layerName = 'conv_2'
# controlObj.curLayerName = layerName
#
# colCount = 8
# margin = 2
#
# import alexnet_utils
# import random
#
# def prepareBatch(imageNums, labels):
#     # import tensorflow as tf
#
#     img_size=(256, 256)
#     crop_size=(227, 227)
#     images = [controlObj.imageDataset.getImage(4, 'source', 'train') \
#               for imageNum in imageNums]
#     # images = [alexnet_utils.imresize(image, img_size) for image in images]
#     # images = np.array(images)
#     # images[:, :, :, 0] -= 123.68
#     # images[:, :, :, 1] -= 116.779
#     # images[:, :, :, 2] -= 103.939
#     print(len(images), images[0].shape, images[0].dtype)
#     rands = np.random.randint(1, high=img_size[0] - crop_size[0], size=(len(images) * 2))
#     randI = 0
#     images2 = []
#     for image in images:
#         image = alexnet_utils.imresize(image, img_size)
# #                         image = next(datagen.flow(np.expand_dims(image, axis=0), batch_size=1))[0]
# #                         image = image[rands[randI] : rands[randI] + crop_size[0],
# #                                       rands[randI + 1] : rands[randI + 1] + crop_size[1], :]
# #                         image[:, :, 0] -= 123
# #                         image[:, :, 1] -= 116
# #                         image[:, :, 2] -= 103
#         images2.append(image)
#         randI += 2
#     print(images2[0].dtype)
#     return np.stack(images2), labels
#
# g_infoCache = DataCache.CDataCache(128)
#
# def getMultWeightsInfo(controlObj, layerName, epochNum):
#     global g_infoCache
#
#     itemCacheName = 'winfo_%s_%d' % (layerName, epochNum)
#     cacheItem = g_infoCache.getObject(itemCacheName)
#     if not cacheItem is None:
#         return cacheItem
#
#     # img = controlObj.imageDataset.getImage(controlObj.curImageNum, 'cropped', 'train')
#     # plt.imshow(img / 255.0)
#     if epochNum != controlObj.netWrapper.curEpochNum:
#         controlObj.netWrapper.loadState(epochNum)
#     weights = controlObj.netWrapper.getMultWeights(layerName, True)
# #     print('%s weights: %s, min %.4f, max %.4f (%s)' % \
# #                 (layerName, str(activations.shape), activations.min(), activations.max(),
# #                  str([int(v[0]) for v in np.where(activations == activations.max())])))
#     info = {'min': weights.min(), 'max': weights.max(), 'stdDev': np.std(weights)}
#     g_infoCache.saveObject(itemCacheName, info)
#     return info
#
# # Debug method for looking at train/test images and predictions for them
# def calcActivations(model, images, classes):
#     import torch
#
#     model.eval()
#     activations = model.forward(images.cuda())
#     # print(activations.shape, activations[:, 70:72])
#     for imgInd, pytImg in enumerate(images[:4]):
#         img = np.transpose(pytImg.numpy(), (1, 2, 0))
#         img = (img + 2.2) / 5
# #         print(img.dtype, img.shape, img.min(), img.max())
#         imshow(img)
#         plt.show()
#         with torch.set_grad_enabled(False):
#             if 1:
#                 activations = model.forward(pytImg.unsqueeze(0).cuda())
#                 npActs = activations[0].cpu().numpy()
#             else:
#                 npActs = activations[imgInd].cpu().numpy()
# #             print(activations.tolist())
#             sortedActInds = npActs.argsort()[::-1]
#             print(activations.shape, classes[imgInd], sortedActInds) #, npActs[sortedActInds])
#
#
# if __name__ == "__main__":
#     print(controlObj.getSelectedEpochNum())
#
#     # activations, drawMode, stdData = controlObj.getActivationsData(epochNum, imageNum, layerName)
#     # actImage = layoutLayersToOneImage(np.sqrt(activations), colCount, margin)
#
#     i1 = controlObj.imageDataset.getImage(4, 'source')
#     i2 = controlObj.imageDataset.getImage(4, 'cropped')
#     i3 = controlObj.imageDataset.getImage(4, 'net')
#     print(prepareBatch(range(1, 9), range(1, 9)))
#
#     controlObj.doCorrelationAnalyzis(False)
#
#     try:
#         for imageNum in range(5, 5):
#             image = controlObj.imageDataset.getImage(imageNum, 'cropped').astype(np.uint8)
#         #     print(image.shape, image.dtype)
#             print('%d - %s' % (controlObj.imageDataset.getImageLabel(imageNum),
#                                controlObj.imageDataset.getClassNameLabel(imageNum)))
#             imshow(image);
#             plt.show()
#
#             activations = controlObj.netWrapper.getImageActivations('dense_3', imageNum, epochNum)
#             print(activations, 'max', activations.max(),
#                   controlObj.imageDataset.getClassNameLabel(int(np.argmax(activations))))
#
#         if 0:     # Reading dataset
#             if 0:
#                 tfDataset = controlObj.imageDataset.getTfDataset()
#                 tfTrainDataset = tfDataset.shuffle(100)
#                 tfTrainDataset = tfTrainDataset.batch(16)
#                 tfTrainDataset = tfTrainDataset.prefetch(2)
#                 # for v in tfTrainDataset[:3]:
#                 #     print(v)
#             else:
#                 controlObj.netWrapper._initMainNet()
#                 trainImageNums = np.arange(1, controlObj.imageDataset.getImageCount('train') + 1)
#                 testImageNums = np.arange(1, controlObj.imageDataset.getImageCount('test') + 1)
#                 tfTrainDataset = controlObj.netWrapper.net._getTfDataset(
#                                 trainImageNums, testImageNums, 1000)
#
#             from itertools import islice
#
#             # for v in list(tfDataset)[:3]:    # Endless
#             for v in islice(tfTrainDataset, 3):
#                 x = v   # x[0].numpy()
#                 print(v[0].shape)
#     except Exception as ex:
#         raise
#         # print("Error: %s" % str(ex))
#
#     controlObj.buildMultActTops()
