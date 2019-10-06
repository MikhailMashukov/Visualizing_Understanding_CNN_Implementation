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
import pickle
import math
import numpy as np
import PyQt4.Qt
from PyQt4 import QtCore, QtGui
# from PySide import QtGui
# import re
    # from glumpy import app, glm   # In case of "GLFW not found" copy glfw3.dll to C:\Python27\Lib\site-packages\PyQt4\
    # from glumpy.transforms import Position, Trackball, Viewport
    # from glumpy.api.matplotlib import *
    # import glumpy.app.window.key as keys
import os
import random
import sys
import time

# sys.path.append(r"../Qt_TradeSim")
import AlexNetVisWrapper
from MyUtils import *
# from ControlWindow import *
# from CppNetChecker import *
# from NeuralNet import *
# from PersistDiagram import *
# from PriceHistory import *


def padImagesToMax(imageList, padValue=255):
    maxSize = 0
    for img in imageList:
        if maxSize < max(img.shape[0:2]):
            maxSize = max(img.shape[0:2])

    resultList = []
    for img in imageList:
        if maxSize != img.shape[0] or maxSize != img.shape[1]:
            padded = np.pad(img, ((0, maxSize - img.shape[0]),
                                  (0, maxSize - img.shape[1]), (0, 0)),
                            constant_values=padValue)
        else:
            padded = img
        resultList.append(padded)
    return resultList

# Transforms e.g. np.array[96, 55, 55] (or [96, 55, 55, 3])
# into image with 10 55 * 55 images horizontally and 9 vertically
def layoutLayersToOneImage(activations, colCount, channelMargin, fillValue=None):
    chanCount = activations.shape[0]
    shift = activations.shape[1] + channelMargin
    if fillValue is None:
        fillValue = 0 if activations.dtype in [np.uint8, np.uint32] else -1
    colMarginData = np.full([activations.shape[2], channelMargin] + list(activations.shape[3:]),
                            fillValue,
                            dtype=activations.dtype)
    rowMarginData = None
    fullList = []
    for layerY in range(chanCount // colCount + 1):
        if (layerY + 1) * colCount > chanCount:
            break
        # rowData = activations[0, layerY * colCount]
        rowList = []
        for layerX in range(colCount):
            if layerX > 0:
                rowList.append(colMarginData)
            rowList.append(activations[layerY * colCount + layerX])

        rowData = np.concatenate(rowList, axis=1)
        if layerY > 0:
            if rowMarginData is None:
                rowMarginData = np.full(
                        [channelMargin, rowData.shape[1]] + list(activations.shape[3:]),
                        fillValue,
                        dtype=activations.dtype)
            fullList.append(rowMarginData)
        fullList.append(rowData)
    return np.concatenate(fullList, axis=0)

# Convolution of multiple channels with one output channel's weights
# and without summing of results
def conv2D_BeforeSumming(activations, weights):
    from scipy import signal

    resultList = []
    for i in range(weights.shape[2]):
        resultList.append(signal.convolve2d(activations[i], weights[:, :, i], \
                boundary='fill', mode='valid', fillvalue=-100))      # 'full'
    return np.stack(resultList, axis=0)

class QtMainWindow(QtGui.QMainWindow): # , DeepMain.MainWrapper):
    c_channelMargin = 2
    c_channelMargin_Top = 5

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        # super(QtMainWindow, self).__init__(parent)
        # DeepMain.MainWrapper.__init__(self, DeepOptions.studyType)
        self.exiting = False
        self.lastAction = None
        self.lastActionStartTime = None
        self.netWrapper = AlexNetVisWrapper.CAlexNetVisWrapper()
        self.imageDataset = self.netWrapper.getImageDataset()
        self.initUI()
        # self.initAlexNetUI()

        # self.showControlWindow()

        self.iterNumLabel.setText('Iteration 0')
        self.maxAnalyzeChanCount = 130

    def init(self):
        # DeepMain.MainWrapper.__init__(self, DeepOptions.studyType)
        # DeepMain.MainWrapper.init(self)
        # DeepMain.MainWrapper.startNeuralNetTraining()
        # self.net = self.netTrader.net
        # self.fastInit()
        self.mousePressPos = None

    def fastInit(self):
        self.curSavedBatchDatasetName = None
        if 1:
            try:
                self.datasetComboBox.addItem("Train")
                self.datasetComboBox.addItem("Test0")
                self.datasetComboBox.addItem("Test1")
                self.datasetComboBox.addItem("Test2")
            except Exception as ex:
                self.infoLabel.setText('Exception in fastInit: %s' % str(ex))
                pass

        self.curDiags = dict()
        self.mousePressPos = None
        self.curInitialBlocks = None
        self.curEditedBlocks = None
        self.prevDrawnResults = None

        try:
            self.plot()
        except Exception as ex:
            self.infoLabel.setText('Exception on plot: %s' % str(ex))
            return
        self.infoLabel.setText('Updated')

    def initUI(self):
        self.setGeometry(100, 40, 1100, 700)
        self.setWindowTitle('Visualization Qt Main')

        c_buttonWidth = 50
        c_buttonHeight = 30
        c_margin = 10

        # set the layout
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)       # Additional _main object - workaround for not appearing FigureCanvas
        layout = QtGui.QVBoxLayout(self._main)
        layout.setContentsMargins(c_margin, c_margin, c_margin, c_margin)
        layout.setSpacing(10)
        # layout.addWidget(self.paramSpinBox)
        self.layout = layout
        # self.setLayout(layout)

        # Widgets line 1 - "Iteration ..."
        y = c_margin
        x = c_margin
        curHorizWidget = QtGui.QHBoxLayout()
        self.iterNumLabel = QtGui.QLabel(self)
        # self.iterNumLabel.setGeometry(x, y, 200, c_buttonHeight)
        curHorizWidget.addWidget(self.iterNumLabel)

        x += 200 + c_margin
        self.datasetComboBox = QtGui.QComboBox(self)
        # self.datasetComboBox.setGeometry(x, y, 200, c_buttonHeight)
        self.datasetComboBox.currentIndexChanged.connect(self.onDatasetChanged)
        curHorizWidget.addWidget(self.datasetComboBox)

        x += 150 + c_margin
        self.blockComboBox = QtGui.QComboBox(self)
        # self.blockComboBox.setGeometry(x, y, 300, c_buttonHeight)
        # curHorizWidget.addWidget(self.blockComboBox)
        # frame = QtGui.QFrame()
        # frame.add
        self.blockComboBox.currentIndexChanged.connect(self.onBlockChanged)
        curHorizWidget.addWidget(self.blockComboBox)

        spinBox = QtGui.QSpinBox(self)
        spinBox.setRange(0, 1023)
        spinBox.setValue(0)
        spinBox.valueChanged.connect(lambda: self.onChanSpinBoxValueChanged())
        curHorizWidget.addWidget(spinBox)
        self.chanNumEdit = spinBox

        button = QtGui.QPushButton('Show &channel act.', self)
        button.clicked.connect(self.onShowChanActivationsPressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Sorted &channel act.', self)
        button.clicked.connect(self.onShowSortedChanActivationsPressed)
        curHorizWidget.addWidget(button)
        layout.addLayout(curHorizWidget)

        # Widgets line 2 - "Mouse move..."
        y += c_buttonHeight + c_margin
        x = c_margin
        curHorizWidget = QtGui.QHBoxLayout()
        self.infoLabel = QtGui.QLabel(self)
        # self.infoLabel.setGeometry(x, y, self.width() - c_margin * 2, c_buttonHeight)
        curHorizWidget.addWidget(self.infoLabel)
        layout.addLayout(curHorizWidget)

        # Widgets line 3
        y += c_buttonHeight + c_margin
        x = c_margin
        curHorizWidget = QtGui.QHBoxLayout()
        button = QtGui.QPushButton('test', self)
        button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        # button.clicked.connect(lambda: self.onTestPress())
        curHorizWidget.addWidget(button)

        x += c_buttonWidth + c_margin
        button = QtGui.QPushButton('1 iteration', self)
        # button.clicked.connect(lambda: self.onDoItersPressed(1))
        curHorizWidget.addWidget(button)

        # lineEdit = QtGui.QLineEdit(self)
        # lineEdit.setValidator(QtGui.QIntValidator(1, 999999))
        # lineEdit.setText("1")
        # curHorizWidget.addWidget(lineEdit)
        # self.iterCountLineEdit = lineEdit   #-
        # self.imageNumLineEdit = lineEdit
        spinBox = QtGui.QSpinBox(self)
        spinBox.setRange(1, 999999)
        spinBox.setValue(1)
        spinBox.valueChanged.connect(lambda: self.onSpinBoxValueChanged())
        curHorizWidget.addWidget(spinBox)
        self.imageNumEdit = spinBox

        button = QtGui.QPushButton('Show &image', self)
        # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        button.clicked.connect(self.onShowImagePressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Show &activations', self)
        button.clicked.connect(self.onShowActivationsPressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Show act. tops', self)
        button.clicked.connect(self.onShowActTopsPressed)
        curHorizWidget.addWidget(button)

        self.multActTopsButtonText = 'Show my &mult. act. tops'
        button = QtGui.QPushButton(self.multActTopsButtonText, self)
        self.multActTopsButton = button
        button.clicked.connect(self.onShowMultActTopsPressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Show all images tops', self)
        button.clicked.connect(self.onShowActTopsFromCsvPressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('I&terations', self)
        # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        button.clicked.connect(lambda: self.onDoItersPressed(int(self.iterCountLineEdit.text())))
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('Load state', self)
        # button.clicked.connect(lambda: self.onLoadStatePressed())
        button = QtGui.QPushButton('Save cache', self)
        button.clicked.connect(lambda: self.saveState())
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('+ learn. rate', self)
        # button.clicked.connect(lambda: self.onIncreaseLearningRatePressed())
        # curHorizWidget.addWidget(button)
        #
        # button = QtGui.QPushButton('Reinit. worst neirons', self)
        # button.clicked.connect(lambda: self.onDeleteWorstNeironsPressed())
        # curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('&Cancel', self)
        button.clicked.connect(self.onCancelPressed)
        curHorizWidget.addWidget(button)
        layout.addLayout(curHorizWidget)

        # Widgets line 4
        if 0:
            curHorizWidget = QtGui.QHBoxLayout()
            lineEdit = QtGui.QLineEdit(self)
            lineEdit.setValidator(QtGui.QIntValidator(1, 99999))
            curHorizWidget.addWidget(lineEdit)
            lineEdit.setText("0")
            self.startSampleIndLineEdit = lineEdit

            lineEdit = QtGui.QLineEdit(self)
            lineEdit.setValidator(QtGui.QIntValidator(1, 99999))
            curHorizWidget.addWidget(lineEdit)
            lineEdit.setText("300")
            self.endSampleIndLineEdit = lineEdit

            lineEdit = QtGui.QLineEdit(self)
            lineEdit.setValidator(QtGui.QIntValidator(111, 999))
            curHorizWidget.addWidget(lineEdit)
            lineEdit.setText("322")
            self.plotNumLineEdit = lineEdit

            button = QtGui.QPushButton('Display', self)
            # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
            button.clicked.connect(lambda: self.onDisplayPressed())
            curHorizWidget.addWidget(button)

            button = QtGui.QPushButton('Display 6 grad.', self)
            button.clicked.connect(lambda: self.onDisplay6Pressed())
            curHorizWidget.addWidget(button)

            button = QtGui.QPushButton('Display interm. res.', self)
            button.clicked.connect(lambda: self.onDisplayIntermResultsPressed())
            # button.mouseMoveEvent.connect(self.mouseMoveEvent)
            # QtCore.QObject.connect(button, QtCore.SIGNAL("clicked()"), self.onDisplayIntermResultsPressed)
            # QtCore.QObject.connect(button, QtCore.SIGNAL("mouseMoveEvent()"), self.mouseMoveEvent)
            curHorizWidget.addWidget(button)
            layout.addLayout(curHorizWidget)

        # Drawings area
        y += c_buttonHeight + c_margin
        x = c_margin
        if 0:
            label = QtGui.QLabel(self)
            label.setGeometry(x, y, self.width() - x - c_margin, self.height() - y - c_margin)

            pixmap = QtGui.QPixmap(label.width(), label.height())
            # QPaintDevice

            self.pixmap = pixmap
            self.imageLabel = label
            label.setPixmap(pixmap)
        else:
            self.figure = Figure(figsize=(5, 3))
            self.canvas = FigureCanvas(self.figure)
            # self.canvas.setGeometry(x, y, self.width() - x - c_margin, self.height() - y - c_margin)
            self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            # self.canvas.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
            # self.canvas.setSexpandingDirections
            # self.oldMouseMoveEvent = self.canvas.mouseMoveEvent
            self.canvas.mpl_connect('motion_notify_event', self.mouseMoveEvent)
            self.canvas.mpl_connect('button_press_event', self.mousePressEvent)
            self.canvas.mpl_connect('button_release_event', self.mouseReleaseEvent)
            # self.canvas.mouseMoveEvent = (lambda event=event: self.mouseMoveEvent(event), self.oldMouseMoveEvent(event))

            self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(self.canvas, self))
            # self.setCentralWidget(self.canvas)
            curHorizWidget.addWidget(self.canvas)
            layout.addWidget(self.canvas)

        self.lastAction = None

        button.setFocus()

    # def initAlexNetUI(self):

        for layerName in self.netWrapper.getNetLayersToVisualize():
            self.blockComboBox.addItem(layerName)

        self.gridSpec = matplotlib.gridspec.GridSpec(2,2, width_ratios=[1,3], height_ratios=[1,1])

        self.blockComboBox.setCurrentIndex(2)
        # self.lastAction = self.onShowActTopsPressed   #d_

    def showProgress(self, str, processEvents=True):
        print(str)
        # self.setWindowTitle(str)
        self.infoLabel.setText(str)
        if processEvents:
            PyQt4.Qt.QCoreApplication.processEvents()

    def startAction(self, actionFunc):
        self.lastAction = actionFunc
        self.lastActionStartTime = datetime.datetime.now()

    def getSelectedImageNum(self):
        return self.imageNumEdit.value()
            # int(self.imageNumLineEdit.text())

    def getSelectedChannelNum(self):
        return self.chanNumEdit.value()

    def getChannelsToAnalyze(self, data):
        if self.maxAnalyzeChanCount and data.shape[0] > self.maxAnalyzeChanCount:
            return data[:self.maxAnalyzeChanCount]
        else:
            return data

    def closeEvent(self, event):
        self.exiting = True

    def onCancelPressed(self):
        self.lastAction = None
        self.showProgress('Cancelling...')

    def onShowImagePressed(self):
        self.startAction(self.onShowImagePressed)
        imageNum = self.getSelectedImageNum()
        imageData = self.imageDataset.getImage(imageNum, 'cropped')
        # mi = imageData.min()
        # ma = imageData.max()
        # for i in range(0, 255, 2):
        #     imageData[i // 2, 50:100, :] = [i, 0, 0]
        #     imageData[i // 2, 105:160, :] = [0, i, 0]
        #     imageData[i // 2, 180:220, :] = [0, 0, i]
        # imageData[:, [1, 4, 7], :] = [[255, 0, 0], [0, 200 ,0], [0, 0, 145]]

        ax = self.figure.add_subplot(self.gridSpec[0, 0])       # GridSpec: [y, x]
        ax.clear()
        ax.imshow(imageData, alpha=1) # , aspect='equal')
        # ax.imshow(imageData, extent=(-100, 127, -100, 127), aspect='equal')
        self.canvas.draw()

    def onShowActivationsPressed(self):
        self.startAction(self.onShowActivationsPressed)
        imageNum = self.getSelectedImageNum()
        imageData = self.imageDataset.getImage(imageNum, 'alexnet')
        layerName = self.blockComboBox.currentText()
        activations = self.netWrapper.getImageActivations(layerName, imageNum)
        # model = alexnet.AlexNet(layerNum, self.alexNet.model)
        # activations = model.predict(self.imageDataset.getImageFilePath(imageNum))
        if len(activations.shape) == 2:   # Dense level scalars
            activations = np.reshape(activations, [activations.shape[1], 1, 1])
            margin = 0
        else:
            activations = self.getChannelsToAnalyze(activations[0])
            margin = self.c_channelMargin

        colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
        data = layoutLayersToOneImage(np.sqrt(activations),
                                      colCount, margin)

        self.figure.set_tight_layout(True)
        ax = self.getMainSubplot()
        ax.clear()
        # plt.subplots_adjust(left=0.01, right=data.shape[0], bottom=0.1, top=0.9)
        ax.imshow(data, cmap='plasma')
        self.canvas.draw()

    def attachCoordinates(self, data):
        shape = data.shape
        arr = np.arange(0, shape[0])
        grid = np.meshgrid(arr, np.arange(0, shape[1]))
            # np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
        coords = np.vstack([grid[0].flatten(), grid[1].flatten(), data.flatten()])
        return coords

    def onShowActTopsPressed(self):
        # import alexnet

        self.startAction(self.onShowActTopsPressed)
        imageNum = self.getSelectedImageNum()
        sourceImageData = self.imageDataset.getImage(imageNum, 'cropped')
        # alexNetImageData = self.imageDataset.getImage(imageNum, 'alexnet')
        layerName = self.blockComboBox.currentText()
        activations = self.netWrapper.getImageActivations(layerName, imageNum)
        activations = self.getChannelsToAnalyze(activations[0])

        sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(layerName)
        if sourceBlockCalcFunc is None:
            return
        colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
        resultList = []
        # if layerName in ['conv_1', 'conv_2']
        #     activations[0, 22:25, 0] = 100     #d_
        for chanInd in range(activations.shape[0]):
            vals = self.attachCoordinates(activations[chanInd])   # Getting list of (x, y, value)
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
        data = layoutLayersToOneImage(data, colCount, self.c_channelMargin)

        ax = self.figure.add_subplot(self.gridSpec[1, 0])
        ax.imshow(data)
        self.canvas.draw()

    class TMultActOpsOptions:
        topCount = 25
        oneImageMaxTopCount = 6
        minDist = 3
        batchSize = 16 * getCpuCoreCount()

    def onShowActTopsFromCsvPressed(self):
        # Fast, based on data produced by activations.py
        self.startAction(self.onShowActTopsFromCsvPressed)
        options = QtMainWindow.TMultActOpsOptions()
        layerNum = self.blockComboBox.currentIndex() + 1
        options.layerNum = layerNum
        self.showActTops_FromCsv(options)

    def onShowMultActTopsPressed(self):
        # My own implementation, from scratch, with images subblocks precision
        self.startAction(self.onShowMultActTopsPressed)
        options = QtMainWindow.TMultActOpsOptions()
        layerNum = self.blockComboBox.currentIndex() + 1
        options.layerNum = layerNum

        self.needShowCurMultActTops = False
        self.multActTopsButton.setText('Save current')
        try:
            self.multActTopsButton.clicked.disconnect()
        except e:
            pass
        self.multActTopsButton.clicked.connect(self.onShowCurMultActTopsPressed)

        # activations = self.getChannelsToAnalyze(self.netWrapper.getImageActivations(layerNum, 1)[0])
        # print(activations)

        bestSourceCoords = None
            # [layerNum][resultNum (the last - the best)] -> (imageNum, x at channel, y, value)
        imageToProcessCount = max(10, self.getSelectedImageNum())
        batchSize = options.batchSize
        prevT = datetime.datetime.now()
        for batchNum in range((imageToProcessCount - 1) // batchSize + 1):
            imageNums = range(batchNum * batchSize + 1, min((batchNum + 1) * batchSize, imageToProcessCount) + 1)
            print("Batch: ", ','.join(str(i) for i in imageNums))
            batchActivations = self.netWrapper.getImagesActivations_Batch(layerNum, imageNums)
            for imageNum in imageNums:
                activations = self.getChannelsToAnalyze(batchActivations[imageNum - imageNums[0]])

                if bestSourceCoords is None:
                    bestSourceCoords = [[] for _ in range(activations.shape[0])]
                # if layerNum <= 2:
                #     activations[0, 22:25, 0] = 100     #d_
                for chanInd in range(activations.shape[0]):
                    vals = self.attachCoordinates(activations[chanInd])    # Getting list of (x, y, value)
                    sortedVals = vals[:, vals[2, :].argsort()]
                    valsToSave = sortedVals[:, -options.oneImageMaxTopCount : ]    # Unfortunately without respect to min. distance
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
                    if self.lastAction is None or self.exiting:   # Cancel or close pressed
                        break
                    elif self.needShowCurMultActTops:
                        self.needShowCurMultActTops = False
                        resultImage = self.showMultActTops(bestSourceCoords, activations.shape[0], options)

                        from scipy.misc import imsave

                        imsave('Data/top%d_conv%d_%dChannels_%dImages.png' %
                                    (options.topCount, layerNum, activations.shape[0], imageNum),
                               resultImage, format='png')

                        # self.figure.savefig('Results/top%d_conv%d_%dChannels_%dImages.png' %
                        #                         (options.topCount, layerNum, activations.shape[0], imageNum),
                        #                     format='png', dpi=resultImageShape[0] / 3)
            if self.lastAction is None or self.exiting:   # Cancel or close pressed
                break
        self.multActTopsButton.setText(self.multActTopsButtonText)
        self.multActTopsButton.clicked.connect(self.onShowMultActTopsPressed)
        if not self.exiting:
            self.showMultActTops(bestSourceCoords, activations.shape[0], options)

    def showMultActTops(self, bestSourceCoords, chanCount, options):
        sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(
                'conv_%d' % options.layerNum)
        if not bestSourceCoords or len(bestSourceCoords[0]) == 0:
            return
        resultList = []
        topColCount = int(math.ceil(math.sqrt(options.topCount)))
        t0 = datetime.datetime.now()
        for chanInd in range(chanCount):
            vals = np.concatenate(bestSourceCoords[chanInd], axis=1)       # E.g. 4 * 100
            if chanInd == 0:
                maxImageNum = np.max(vals[0, :])
            sortedVals = vals[:, vals[3, :].argsort()]

            selectedImageList = []
            selectedList = []                   # Will be e.g. 9 * 4
            i = sortedVals.shape[1] - 1
            while len(selectedList) < options.topCount and i >= 0:
                curVal = sortedVals[:, i]
                isOk = True
                for prevVal in selectedList:
                    if curVal[0] == prevVal[0] and abs(curVal[1] - prevVal[1]) < options.minDist and \
                            abs(curVal[2] - prevVal[2]) < options.minDist:
                        isOk = False
                        break
                if isOk:
                    sourceBlock = sourceBlockCalcFunc(int(curVal[1]), int(curVal[2]))
                    imageData = self.imageDataset.getImage(int(curVal[0]), 'cropped')
                    selectedImageList.append(imageData[sourceBlock[0] : sourceBlock[2], sourceBlock[1] : sourceBlock[3], :])
                    selectedList.append(curVal)
                i -= 1
            selectedImageList = padImagesToMax(selectedImageList)
            chanData = np.stack(selectedImageList, axis=0)
            chanImageData = layoutLayersToOneImage(chanData, topColCount, 1, 255)
            resultList.append(chanImageData)
            bestSourceCoords[chanInd] = [np.stack(selectedList).transpose()]
            if (chanInd + 1) % 4 == 0:
                t = datetime.datetime.now()
                if (t - t0).total_seconds() >= 1:
                    self.showProgress('Stage 2: %d channels' % \
                                      (chanInd + 1))
                    t0 = t

        resultList = padImagesToMax(resultList)
        data = np.stack(resultList, axis=0)
        colCount = math.ceil(math.sqrt(chanCount) * 1.15 / 2) * 2
        data = layoutLayersToOneImage(data, colCount, self.c_channelMargin_Top)

        # try:
        #     figure, axes = plt.subplots(223)
        #     figure.delaxes(axes)
        #     figure, axes = plt.subplots(224)
        #     figure.delaxes(axes)
        # except Exception as ex:
        #     print('Exception on subplot deletion: %s' % str(ex))
        ax = self.getMainSubplot()
        ax.clear()
        ax.imshow(data)
        self.canvas.draw()

        import pickle

        fileName = 'Data/BestActs%d_conv%d_%dImages.dat' % \
                (options.topCount, options.layerNum, int(maxImageNum))
        with open(fileName, 'wb') as file:
            pickle.dump(bestSourceCoords, file)
        return data

    def showActTops_FromCsv(self, options):
        actMatrix = self.netWrapper.getImagesActivationMatrix(options.layerNum)
        chanCount = actMatrix.shape[1]
        if self.maxAnalyzeChanCount and chanCount > self.maxAnalyzeChanCount:
            chanCount = self.maxAnalyzeChanCount
        sortedValsList = []
        for chanInd in range(chanCount):
            sortedImageNums = np.flip(actMatrix[:, chanInd].argsort())[:options.topCount] + 1
            sortedValsList.append(sortedImageNums)
        imageNums = np.stack(sortedValsList, axis=0)
        self.showTopImages(imageNums, options)

    def showTopImages(self, imageNums, options):
        topColCount = int(math.ceil(math.sqrt(options.topCount)))
        chanCount = imageNums.shape[0]
        resultList = []
        t0 = datetime.datetime.now()
        for chanInd in range(chanCount):
            selectedImageList = []
            for imageNum in imageNums[chanInd][:options.topCount]:
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
        data = layoutLayersToOneImage(data, colCount, self.c_channelMargin_Top)

        ax = self.getMainSubplot()
        ax.clear()
        ax.imshow(data)
        self.canvas.draw()
        return data.shape[0:2]


    def onShowCurMultActTopsPressed(self):
        self.needShowCurMultActTops = True

    def onShowChanActivationsPressed(self):
        try:
            self.startAction(self.onShowChanActivationsPressed)
            imageNum = self.getSelectedImageNum()
            layerNum = self.blockComboBox.currentIndex() + 1
            chanNum = self.getSelectedChannelNum()
            activations = self.netWrapper.getImageActivations(layerNum, imageNum)
            activations = activations[0][chanNum]

            self.figure.clear()
            self.mainSubplotAxes = None
            self.figure.set_tight_layout(True)

            showCurLayerActivations = True
            if showCurLayerActivations:
                imageData = np.sqrt(activations)
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.imshow(imageData, cmap='plasma')

            vals = self.attachCoordinates(activations)        # Getting list of (x, y, value)
            sortedVals = vals[:, vals[2, :].argsort()]
            curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))

            # ax = self.figure.add_subplot(self.gridSpec[0, 0])
            # ax.clear()
            # ax.hist([1,2,3], bins=100)

            # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
            # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
            allWeights = self.netWrapper.getNetWeights('conv_%d' % layerNum)
            weights = allWeights[0].numpy()[:, :, :, chanNum]
            if layerNum == 1:
                prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
            else:
                prevLayerActivations = self.netWrapper.getImageActivations(layerNum - 1, imageNum)[0]
                        # Conv2 - 256 * 27 * 27

            if not showCurLayerActivations:
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.hist(weights.flatten(), bins=100)
                if ax.get_ylim()[0] > ax.get_ylim()[1]:
                    ax.invert_yaxis()
                ax.set_aspect('auto')

            colCount = math.ceil(math.sqrt(weights.shape[2]) * 1.15 / 2) * 2
            self.showWeights(self.figure.add_subplot(self.gridSpec[1, 0]), weights, colCount)

            # bias = allWeights[1].numpy()[chanNum]
            chanConvData = conv2D_BeforeSumming(prevLayerActivations, weights)
            convImageData = layoutLayersToOneImage(chanConvData, colCount, 1, chanConvData.min())
            ax = self.getMainSubplot()
            # ax = self.figure.add_subplot(self.gridSpec[1, 0])
            # ax = self.figure.add_subplot(self.gridSpec[:, 1])
            ax.clear()
            im = ax.imshow(convImageData, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)
            # self.figure.get_colorbar(ax=ax)
            self.canvas.draw()
        except Exception as ex:
            self.showProgress("Error: %s" % str(ex))

    def onShowSortedChanActivationsPressed(self):
        try:
            self.startAction(self.onShowSortedChanActivationsPressed)
            imageNum = self.getSelectedImageNum()
            layerNum = self.blockComboBox.currentIndex() + 1
            chanNum = self.getSelectedChannelNum()
            activations = self.netWrapper.getImageActivations(layerNum, imageNum)
            activations = activations[0][chanNum]

            self.figure.clear()
            self.mainSubplotAxes = None
            self.figure.set_tight_layout(True)

            showCurLayerActivations = True
            if showCurLayerActivations:
                imageData = np.sqrt(activations)
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.imshow(imageData, cmap='plasma')

            vals = self.attachCoordinates(activations)        # Getting list of (x, y, value)
            sortedVals = vals[:, vals[2, :].argsort()]
            curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))

            # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
            # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
            allWeights = self.netWrapper.getNetWeights('conv_%d' % layerNum)
            weights = allWeights[0].numpy()[:, :, :, chanNum]
            if layerNum == 1:
                prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
            else:
                prevLayerActivations = self.netWrapper.getImageActivations(layerNum - 1, imageNum)[0]
                        # Conv2 - 256 * 27 * 27

            if not showCurLayerActivations:
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.hist(weights.flatten(), bins=100)
                if ax.get_ylim()[0] > ax.get_ylim()[1]:
                    ax.invert_yaxis()
                ax.set_aspect('auto')

            chanCount = weights.shape[2]
            showChanCount = int(math.ceil(chanCount / 4))
            colCount = math.ceil(math.sqrt(showChanCount) * 1.15 / 2) * 2

            # bias = allWeights[1].numpy()[chanNum]
            chanConvData = conv2D_BeforeSumming(prevLayerActivations, weights)
            # vals = np.stack([np.arange(chanConvData.shape[0]),
            #                  np.max(np.abs(chanConvData), axis=(1,2))], axis=0)
            # prevChansOrder = np.flip(vals[1, :].argsort())
            vals = np.max(np.abs(chanConvData), axis=(1,2))
            prevChansOrder = np.flip(vals.argsort())
            sortedVals = chanConvData[prevChansOrder, :, :]
            convImageData = layoutLayersToOneImage(sortedVals[:showChanCount], colCount, 1, chanConvData.min())
            # ax = self.getMainSubplot()
            ax = self.figure.add_subplot(self.gridSpec[0, 0])
            ax.clear()
            im = ax.imshow(convImageData, cmap='plasma')
            # colorBar = self.figure.colorbar(im, ax=ax)

            options = QtMainWindow.TMultActOpsOptions()
            options.layerNum = layerNum
            actMatrix = self.netWrapper.getImagesActivationMatrix(layerNum)
            sortedValsList = []
            for chanInd in range(showChanCount):
                sortedImageNums = np.flip(actMatrix[:, prevChansOrder[chanInd]].argsort())[:options.topCount] + 1
                sortedValsList.append(sortedImageNums)
            imageNums = np.stack(sortedValsList, axis=0)
            self.showTopImages(imageNums, options)

            self.showWeights(self.figure.add_subplot(self.gridSpec[1, 0]), weights[:, :, prevChansOrder],
                             colCount * 2)

            self.canvas.draw()
        except Exception as ex:
            self.showProgress("Error: %s" % str(ex))

    def showWeights(self, ax, weights, colCount):
        weightsImageData = weights.transpose((2, 0, 1))
        weightsImageData = layoutLayersToOneImage(weightsImageData, colCount, 1, weights.min())

        ax.clear()
        ax.imshow(weightsImageData, cmap='plasma')


    def onChanSpinBoxValueChanged(self):
        if self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed]:
            self.lastAction()

    def onSpinBoxValueChanged(self):
        if self.lastAction in [self.onShowImagePressed, self.onShowActivationsPressed, self.onShowActTopsPressed]:
            t0 = datetime.datetime.now()
            self.onShowActTopsPressed()
            self.onShowImagePressed()
            self.onShowActivationsPressed()
            self.showProgress('3 operations: %.1f ms' % ((datetime.datetime.now() - t0).total_seconds() * 1000))

            # self.lastAction()
        elif self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed]:
            self.lastAction()

    def onBlockChanged(self):
        if self.lastAction in [self.onShowImagePressed, self.onShowActivationsPressed, self.onShowActTopsPressed]:
            self.onSpinBoxValueChanged()

    def getMainSubplot(self):
        if not hasattr(self, 'mainSubplotAxes') or self.mainSubplotAxes is None:
            self.mainSubplotAxes = self.figure.add_subplot(self.gridSpec[:, 1])
        return self.mainSubplotAxes


    def saveState(self):
        try:
            self.netWrapper.saveCacheState()
        except Exception as ex:
            self.showProgress("Error in saveState: %s" % str(ex))

    def loadState(self):
        try:
            self.netWrapper.loadCacheState()
        except Exception as ex:
            self.showProgress("Error in loadState: %s" % str(ex))


    def mouseMoveEvent(self, event):
        try:
            s = 'Mouse move: %d, %d' % (int(event.x), int(event.y))
            self.infoLabel.setText(s)
            return

            editBlockInd = None
            if event.inaxes and event.inaxes.get_navigate():
                try:
                    s = event.inaxes.format_coord(event.xdata, event.ydata)
                    if event.inaxes in self.curDiags:
                        editBlockInd = self.curDiags[event.inaxes]
                    else:
                        editBlockInd = -1
                    s = 'Block %d, %s' % (editBlockInd, s)
                except:
                    pass

                else:
                    artists = [a for a in event.inaxes.mouseover_set
                               if a.contains(event) and a.get_visible()]
                    if artists:
                        a = max(artists, key=lambda x: x.zorder)
                        if a is not event.inaxes.patch:
                            data = a.get_cursor_data(event)
                            if data is not None:
                                s += ' [%s]' % a.format_cursor_data(data)

                    # if len(self.mode):
                    #     s = '%s, %s' % (self.mode, s)
                    # else:
            self.infoLabel.setText('Mouse move: %s' % s)

            if event.button == 1:
                self.onDiagMouseChange(editBlockInd, event.xdata, event.ydata, event.y)
        except Exception as ex:
            print("Exception in mouseMove: ", str(ex))

    def mousePressEvent(self, event):
        return
        if event.inaxes and event.inaxes.get_navigate():
            try:
                x = int(event.xdata)
                y = int(event.ydata)
                if event.inaxes in self.curDiags:
                    editBlockInd = self.curDiags[event.inaxes]
                else:
                    editBlockInd = -1
                self.mousePressPos = (editBlockInd, event.xdata, event.ydata, event.y)
            except:
                pass
        else:
            self.infoLabel.setText('Mouse press: no data selected')

    def onDiagMouseChange(self, editBlockInd, x, y, screenY):
        if self.mousePressPos is None:
            self.mousePressPos = (editBlockInd, x, y, screenY, self.getClosestY(editBlockInd, x, y))
            return
        editBlockInd = self.mousePressPos[0]
        x = int(self.mousePressPos[1] + 0.5)
        editElemY = self.mousePressPos[4]
        sampleBlock = self.curEditedBlocks[editBlockInd + 1]
        blockShape = sampleBlock.shape
        print("blockShape %s, start %s, x %d, screen y %d, selected y %d" %
              (str(blockShape), str(self.mousePressPos), x, screenY, editElemY))
        if x < 0:
            x = 0
        if x >= blockShape[1]:
            x = blockShape[1] - 1
        if editElemY < 0:
            editElemY = 0
        if editElemY >= blockShape[0]:
            editElemY = blockShape[0] - 1

        self.curEditedBlocks[editBlockInd + 1][editElemY][x] = \
                self.curInitialBlocks[editBlockInd + 1][editElemY][x] + \
                (screenY - self.mousePressPos[3]) / 300

        print("V ", self.curEditedBlocks[editBlockInd + 1][editElemY][x])

        # self.figure.clear()
        ax = self.figure.add_subplot(331)
        ax.clear()
        ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[0], color=(0.3, 0.3, 0.8, 0.5), linestyle = 'dashed')
        ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[1], color=(0.7, 0.3, 0.3, 0.5))
        ax.plot(np.arange(sampleBlock.shape[1]), sampleBlock[2], color=(0.3, 0.7, 0.3, 0.5))
        self.canvas.draw()

    def getClosestY(self, editBlockInd, x, y):
        intX = int(x + 0.5)
        sampleBlock = self.curEditedBlocks[editBlockInd + 1]
        minDist = None
        minDistElemY = None
        for elemY in range(sampleBlock.shape[0]):
            dist = abs(sampleBlock[elemY][intX] - y)
            print("Dist %d: %f - %f" % (elemY, sampleBlock[elemY][intX], y))
            if elemY == 0 or minDist > dist:
                minDist = dist
                minDistElemY = elemY
        return minDistElemY

    def mouseReleaseEvent(self, event):
        if self.mousePressPos is None:
            return

        self.mousePressPos = None
        return

        samples = self.getSamples()
        if not samples:
            return
        sample = (np.expand_dims(self.curEditedBlocks[0].transpose(), 0),
                  samples[1][:1, :], samples[2][:1, :])
        feedDict = self.netTrader.getNetFeedDict_SamplesRange('train', sample)

        newResults = [sample]
        for intermBlockInd in range(self.net.getIntermResultsBlockCount()):
            block = self.net.getIntermResultsBlock(intermBlockInd, feedDict)
            if self.prevDrawnResults is not None:
                self.drawSamplesIntermResults(self.prevDrawnResults[intermBlockInd + 1],
                                              intermBlockInd, 332 + intermBlockInd, True)
            self.drawSamplesIntermResults(block, intermBlockInd, 332 + intermBlockInd,
                                          False, self.prevDrawnResults is None)
                    # int(self.plotNumLineEdit.text()))
            self.curInitialBlocks.append(block)
            newResults.append(block)
        self.prevDrawnResults = newResults
        self.infoLabel.setText('Results updated')

    def plotSamplesResults(self, testSamplesActuals, testSamplesPreds):
        import random

        if 0:
            ''' plot some random stuff '''

            self._static_ax = self.canvas.figure.subplots()
            t = np.linspace(0, 10, 501)
            self._static_ax.plot(t, np.tan(t), ".")


        # data = [random.random() for i in range(10)]

        (testSamplesActuals, testSamplesPreds) = self.sortByClass(testSamplesActuals, testSamplesPreds)
        (_, testActualClassNums) = np.where(testSamplesActuals==1)

        ax = self.figure.add_subplot(111)
        ax.clear()
        # ax.plot(data, '*-')
        ax.matshow(np.matrix.transpose(testSamplesPreds), aspect='auto', cmap="YlGn")
        # ax.matshow(np.matrix.transpose(testSamplesActuals), aspect='auto', \
        #            cmap="YlGn", hatch='/', alpha=0.5)
        del self.curDiags[ax.axes]
        ax.scatter(np.arange(testSamplesActuals.shape[0]), testActualClassNums, alpha=0.5)
        self.canvas.draw()

    def drawSamplesResults(self, datasetName):
        if self.net:
            samples = self.getSamples()
            # feedDict = self.netTrader.getNetFeedDict(datasetName.lower(), 300)
            feedDict = self.netTrader.getNetFeedDict_SamplesRange(datasetName, samples)
            (samplesActuals, samplesPreds) = self.net.getSamplesResults(feedDict)
            for otherNet in self.otherNets:
                (otherActuals, otherPreds) = otherNet.getSamplesResults(feedDict)
                if not np.array_equal(samplesActuals, otherActuals):
                    print("samplesActuals != otherActuals")
                samplesPreds = np.concatenate((samplesPreds, otherPreds), axis=1)
        else:  # Fast init
            (samplesActuals, samplesPreds) = self.createSamplesResultsTestData()

        self.plotSamplesResults(samplesActuals, samplesPreds)
        self.infoLabel.setText('Displayed data for dataset %s' % datasetName)

    def drawGradients(self, datasetName, gradBlockInd, plotNum = 111, \
                                 startSampleInd = None, endSampleInd = None):
        samples = self.getSamples(startSampleInd, endSampleInd)
        # feedDict = self.netTrader.getNetFeedDict(datasetName.lower(), 300)
        feedDict = self.netTrader.getNetFeedDict_SamplesRange(datasetName, samples)
        block = self.net.getGradientsBlock(gradBlockInd, feedDict)
        self.infoLabel.setText('Block size: %s' % str(block.shape))
        block = np.squeeze(block)

        cdict = {
          'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
          'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
          'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
        }

        if 0:
            cm = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

            x = np.arange(0, 10, .1)
            y = np.arange(0, 10, .1)
            X, Y = np.meshgrid(x,y)

            data = 2*( np.sin(X) + np.sin(3*Y) )
            f =  lambda x:np.clip(x, -4, 0)

        ax = self.figure.add_subplot(plotNum)
        ax.clear()
        # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)
        # ax.plot(data, '*-')
        while len(block.shape) > 2:
            block = np.squeeze(block[0])
        self.im = ax.matshow(np.matrix.transpose(block), aspect='auto', cmap="YlGn")
        self.curDiags[ax.axes] = gradBlockInd      # TODO? to replace with del self.curDiags[ax.axes]
        # ax.matshow(np.matrix.transpose(testSamplesActuals), aspect='auto', \
        #            cmap="YlGn", hatch='/', alpha=0.5)
        # scatter = plt.scatter([1, 2], [30, 4], cmap="YlGn")
        # scatter.set_array(np.array([block.min(), block.max()]))
        # self.figure.colorbar(scatter)
        # plt.colorbar()

        print("Stddev %f, mean %f" % (np.std(block), np.mean(block)))
        if np.std(block) > 0.0015:
            mult = 1000
        else:
            mult = 10000

        absMeanByInput = np.mean(abs(block), axis=1)
        absMeanByOutput = np.mean(abs(block), axis=0)
        # font = {'fontname':'Arial', 'size':'7'}
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                multV = block[i, j] * mult
                if abs(multV) >= 2:
                    text = ax.text(i, j, int(multV),
                                   ha="center", va="center", color="w", fontsize=7)
            ax.text(i, block.shape[1] + 0.5, int(absMeanByInput[i] * 1e5),
                    ha="center", va="center", color="b", fontsize=7)
            # print("Mean abs: ", np.mean(abs(blockGradients), axis=1))
        for j in range(block.shape[1]):
            ax.text(block.shape[0] + 0.5, j, int(absMeanByOutput[j] * 1e5),
                    ha="center", va="center", color="b", fontsize=7)

        if gradBlockInd == 0:
            (samplesActuals, _) = self.net.getSamplesResults(feedDict)
            (_, actualClassNums) = np.where(samplesActuals==1)
            ax.scatter(np.arange(actualClassNums.shape[0]), actualClassNums, alpha=0.5)

        self.canvas.draw()

    def drawSamplesIntermResults(self, block, intermBlockInd, plotNum = 111,
                                 arePrevResults = False, clearExisting = True):
        self.infoLabel.setText('Block size: %s' % str(block.shape))

        ax = self.figure.add_subplot(plotNum)
        if clearExisting:
            ax.clear()
        block = np.squeeze(block)
        if len(block.shape) == 1:
            ax.plot(np.arange(block.shape[0]), block, '*-', \
                    color=((0, 0.3, 0.6, 0.4) if arePrevResults else (0, 0, 0.8, 0.7)), \
                    linestyle=(':' if arePrevResults else '-'))
        else:
            # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)
            # ax.plot(data, '*-')
            if arePrevResults:
                return
            self.im = ax.matshow(np.matrix.transpose(block), aspect='auto', cmap="YlGn")
        self.curDiags[ax.axes] = intermBlockInd

        print("Stddev %f, mean %f" % (np.std(block), np.mean(block)))
        if 0:
            if np.std(block) > 0.0015:
                mult = 1000
            else:
                mult = 10000

            absMeanByInput = np.mean(abs(block), axis=1)
            absMeanByOutput = np.mean(abs(block), axis=0)
            # font = {'fontname':'Arial', 'size':'7'}
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    multV = block[i, j] * mult
                    if abs(multV) >= 2:
                        text = ax.text(i, j, int(multV),
                                       ha="center", va="center", color="w", fontsize=7)
                ax.text(i, block.shape[1] + 0.5, int(absMeanByInput[i] * 1e5),
                        ha="center", va="center", color="b", fontsize=7)
                # print("Mean abs: ", np.mean(abs(blockGradients), axis=1))
            for j in range(block.shape[1]):
                ax.text(block.shape[0] + 0.5, j, int(absMeanByOutput[j] * 1e5),
                        ha="center", va="center", color="b", fontsize=7)

        self.canvas.draw()

    def onDisplayPressed(self):
        # self.onDatasetChanged(self.datasetComboBox.currentIndex())
        datasetName = self.datasetComboBox.itemText(self.datasetComboBox.currentIndex())
        try:
            blockInd = self.blockComboBox.currentIndex()
            blockName = self.blockComboBox.itemText(blockInd)
            # result = re.search('Gradient.* (\d+)\. ', blockName)
            # if result:
            #     self.drawGradients(datasetName, int(result.group(1)), int(self.plotNumLineEdit.text()))
            # elif blockName == "Samples results":
            #     self.drawSamplesResults(datasetName)
        except Exception as ex:
            print('Exception in onDisplayPressed: %s' % str(ex))

    def onDisplayIntermResultsPressed(self):
        samples = self.getSamples()
        if not samples:
            return
        sample = (samples[0][:1, :, :], samples[1][:1, :], samples[2][:1, :])
        feedDict = self.netTrader.getNetFeedDict_SamplesRange('train', sample)

        # intermBlockInd = self.blockComboBox.currentIndex()
        # if hasattr(sample, 'realDiffs'):
        #         plt.plot(sample.realDiffs, color=(0, 0.8, 0))
        self.figure.clear()
        ax = self.figure.add_subplot(331)
        ax.clear()
        # if self.prevDrawnResults is not None:
        #     ax.plot(np.arange(sample[1].shape[1]), self.prevDrawnResults[0][1][0, :], color=(0.3, 0.3, 0.8, 0.3), linestyle = ':')
        #     ax.plot(np.arange(sample[0].shape[1]), self.prevDrawnResults[0][0][0, :, 1], color=(0.7, 0.3, 0.3, 0.3), linestyle = ':')
        #     ax.plot(np.arange(sample[0].shape[1]), self.prevDrawnResults[0][0][0, :, 2], color=(0.3, 0.7, 0.3, 0.3), linestyle = ':')
        ax.plot(np.arange(sample[1].shape[1]), sample[1][0, :], color=(0.3, 0.3, 0.8, 0.65), linestyle = 'dashed')
        ax.plot(np.arange(sample[0].shape[1]), sample[0][0, :, 1], color=(0.7, 0.3, 0.3, 0.65))
        ax.plot(np.arange(sample[0].shape[1]), sample[0][0, :, 2], color=(0.3, 0.7, 0.3, 0.65))
        print("Sample %s %s" % (str(sample[0].shape), str(sample[1].shape)))
        self.curInitialBlocks = [sample[0][0].transpose()] #np.stack([sample[1][0, :], sample[0][0, :, 1], sample[0][0, :, 2]])]

        newResults = [sample]
        #d_ for intermBlockInd in range(self.net.getIntermResultsBlockCount()):
        #     block = self.net.getIntermResultsBlock(intermBlockInd, feedDict)
        #     self.drawSamplesIntermResults(block, intermBlockInd, 332 + intermBlockInd)
        #             # int(self.plotNumLineEdit.text()))
        #     self.curInitialBlocks.append(block)
        #     newResults.append(block)
        self.curEditedBlocks = [copy.deepcopy(arr) for arr in self.curInitialBlocks]
        self.prevDrawnResults = newResults

        sampleInd = int(self.startSampleIndLineEdit.text())
        self.startSampleIndLineEdit.setText(str(int(sampleInd + 1)))

    def onDatasetChanged(self, datasetInd):
        self.onDisplayPressed()

        # datasetName = self.datasetComboBox.itemText(datasetInd)
        # self.drawSamplesResults(datasetName)

    def onTestPress(self):
        # self.plot()
        # self.onDatasetChanged(0)

        if 0:
            pixmap2 = QtGui.QPixmap(self.imageLabel.width(), self.imageLabel.height())
            painter = QtGui.QPainter(pixmap2)
            painter.setBrush(QtCore.Qt.green)
            painter.drawRect(20, 10, 850, 1130)
            painter.end()

            painter = QtGui.QPainter(self.pixmap)
            painter.setBrush(QtCore.Qt.white)
            painter.drawRect(20, 50, 1050, 1130)
            painter.end()
            # self.pixmap.setp
            for i in range(1000):
                self.imageLabel.setPixmap(self.pixmap)
                self.imageLabel.setPixmap(pixmap2)
                # self.imageLabel.render()

        if 0:
            pixmap = QtGui.QPixmap.grabWindow(
                    # QtGui.QApplication.desktop().winId(),
                    self.winId(),
                    x=0, y=0, width=1000, height=500)

            image = pixmap.toImage()
            for x in range(300):
                for y in range(300):
                    c = QtGui.QColor(image.pixel(x + 25, y))
                    c.setRed(255 if (c.red() + x) >= 255 else (c.red() + x))

                    image.setPixel(x + 25, y, c.rgb())
            pixmap = QtGui.QPixmap().fromImage(image)
            # self.imageLabel.setPixmap(pixmap)
            pass

    def onDoItersPressed(self, iterCount):
        epochIterCount = DeepOptions.trainSetSize / DeepOptions.batchSize
        c_callIterCount = int(epochIterCount)  # 400

        restIterCount = iterCount
        while restIterCount > 0:
            learnResult = self.netTrader.doLearning(
                    c_callIterCount if restIterCount > c_callIterCount else restIterCount)
            restIterCount -= c_callIterCount
            trainIterNum = self.netTrader.getTrainIterNum()
            self.iterNumLabel.setText('Iteration %d' % trainIterNum)
            if self.exiting:
                return
            app.processEvents()

            epochNum = trainIterNum / epochIterCount
            print("Epoch * 3 %%: ", (epochNum * 3) % (DeepOptions.learningRate.circularStepEpochCount * 3))
            if int(epochNum) % (DeepOptions.learningRate.circularStepEpochCount * 3) == 0:
                print("Saving state after %d epochs (%d iterations)" % (epochNum, trainIterNum))
                DeepMain.MainWrapper.saveState(self, "States/State_%d_%05d_%f/state.dat" % \
                        (epochNum, trainIterNum, learnResult))

        self.onDisplayPressed()

    def onIncreaseLearningRatePressed(self):
        self.net.changeRate(2)
        self.infoLabel.setText('Learning rate increased')

    def onLoadStatePressed(self):
        if os.path.isfile(self.stateFileName + '.index'):
            try:
               # and self.studyType != StudyType.DeepDiff:
                self.loadState(self.stateFileName)
                self.onDisplayPressed()
                self.infoLabel.setText('State loaded from %s' % self.stateFileName)
            except Exception as ex:
                self.infoLabel.setText('Exception on state loading: %s' % str(ex))
                print('Exception on state loading: %s' % str(ex))

    def onDeleteWorstNeironsPressed(self):
        samples = self.getSamples(0, 10000)
        feedDict = self.netTrader.getNetFeedDict_SamplesRange('train', samples)
        # block = self.net.getGradientsBlock(2, feedDict)
        self.net.reinitWorstNeirons(2, feedDict)
        self.net.reinitWorstNeirons(3, feedDict)
        self.net.reinitWorstNeirons(4, feedDict)
        self.infoLabel.setText('Worst neirons reinitialized')

    def getSamples(self, startSampleInd = None, endSampleInd = None):
        datasetName = self.datasetComboBox.itemText(self.datasetComboBox.currentIndex())
        if self.curSavedBatchDatasetName != datasetName:
            dataset = self.netTrader.getDatasetByName(datasetName.lower())
            sampleToSaveCount = 10000
            if sampleToSaveCount > len(dataset):
                sampleToSaveCount = len(dataset)
            self.curSavedBatch = dataset.getBatch(sampleToSaveCount)
            print("Loaded %d samples from %s" % (len(self.curSavedBatch[0]), datasetName))
            self.curSavedBatchDatasetName = datasetName
        if startSampleInd is None:
            startSampleInd = int(self.startSampleIndLineEdit.text())
        if endSampleInd is None:
            endSampleInd = int(self.endSampleIndLineEdit.text())
        return tuple([arr[startSampleInd : endSampleInd] for arr in self.curSavedBatch])

    def loadData(self):
        # with open('Block.dat', "rb") as file:
        #     self.block = pickle.load(file)
        pass









    def initWindow(self):
        # self.window = window
        self.frameNumber = 0  # Only used in set_position workaround
        # self.transform = Trackball(Position())
        # self.viewport = Viewport()
        # self.viewport.clipping = False
        # self.diagramOptions = DiagramOptions()
        # self.diagramOptions.xDimInd = 0
        # self.diagramOptions.xDimInd = 1
        # self.diagramOptions.step = 0.2
        # self.diagramOptions.stepCount = 40
        # # self.lastOperation = None
        # self.diagramDisplayMethod = self.displayAllWeightsDiagrams
        #
        # self.window.attach(self.transform)
        # self.window.attach(self.viewport)


    # def showControlWindow(self):
    #     self.controlWin = ControlWindow(window, self)
    #     self.showCurParameters()
    #     self.controlWin.show()


    # def takeScreenShot(self):
    #     """ Takes a screenshot of the screen at give pos & size (rect). """
    #     print('Taking screenshot...')
    #     rect = self.GetRect()
    #     # see http://aspn.activestate.com/ASPN/Mail/Message/wxpython-users/3575899
    #     # created by Andrea Gavana
    #
    #     # Create a DC for the whole screen area
    #     dcScreen = wx.ScreenDC()
    #
    #     # Create a Bitmap that will hold the screenshot image later on
    #     # Note that the Bitmap must have a size big enough to hold the screenshot
    #     # -1 means using the current default colour depth
    #     bmp = wx.EmptyBitmap(rect.width, rect.height)
    #
    #     # Create a memory DC that will be used for actually taking the screenshot
    #     memDC = wx.MemoryDC()
    #
    #     # Tell the memory DC to use our Bitmap
    #     # all drawing action on the memory DC will go to the Bitmap now
    #     memDC.SelectObject(bmp)
    #
    #     # Blit (in this case copy) the actual screen on the memory DC
    #     # and thus the Bitmap
    #     memDC.Blit(0,  # Copy to this X coordinate
    #                0,  # Copy to this Y coordinate
    #                rect.width,  # Copy this width
    #                rect.height,  # Copy this height
    #                dcScreen,  # From where do we copy?
    #                rect.x,  # What's the X offset in the original DC?
    #                rect.y  # What's the Y offset in the original DC?
    #                )
    #
    #     # Select the Bitmap out of the memory DC by selecting a new
    #     # uninitialized Bitmap
    #     memDC.SelectObject(wx.NullBitmap)
    #
    #     img = bmp.ConvertToImage()
    #     fileName = "myImage.png"
    #     img.SaveFile(fileName, wx.BITMAP_TYPE_PNG)

    def createAxes(self):
        self.ticks = SegmentCollection(mode="agg", transform=self.transform, viewport=self.viewport,
                                       linewidth='local', color='local')

        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
        z = 0
        # Frame
        P0 = [(xmin, ymin, z), (xmin, ymax, z), (xmax, ymax, z), (xmax, ymin, z)]
        P1 = [(xmin, ymax, z), (xmax, ymax, z), (xmax, ymin, z), (xmin, ymin, z)]
        self.ticks.append(P0, P1, linewidth=2)

        # Grids
        n = 11
        P0 = np.zeros((n - 2, 3))
        P1 = np.zeros((n - 2, 3))

        P0[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
        P0[:, 1] = ymin
        P0[:, 2] = z
        P1[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
        P1[:, 1] = ymax
        P1[:, 2] = z
        self.ticks.append(P0, P1, linewidth=1, color=(0, 0, 0, .25))

        P0 = np.zeros((n - 2, 3))
        P1 = np.zeros((n - 2, 3))
        P0[:, 0] = xmin
        P0[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
        P0[:, 2] = z
        P1[:, 0] = xmax
        P1[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
        P1[:, 2] = z
        self.ticks.append(P0, P1, linewidth=1, color=(0, 0, 0, .25))

    def createPoints(self):
        # Create a new collection of points
        self.points = PointCollection(mode="agg", transform=self.transform, viewport=self.viewport, color="local")

        # Add a view of the collection on the left subplot
        # left.add_drawable(points)

        # # Change xscale range on left subplot
        # left.transform['zscale']['range'] = -0.5,+0.5
        # # Set trackball view
        # left.transform['trackball']["phi"] = 0
        # left.transform['trackball']["theta"] = 0

        # Add some points
        # self.points.append(np.random.normal(0.0, 0.75, (100000, 3)))
        # self.points.append(np.random.uniform(-0.5, 0.5, (10000, 3)))

        # Show figure
        # figure.show()

    def showCurParameters(self):
        params = np.zeros(self.neuralNet.getParamCount(), np.double)
        self.neuralNet.getParamValues(params)
        self.controlWin.setParamValues(params)

    def displaySourceData(self, dataset):
        input = dataset.data['input']
        target = dataset.data['target']
        # coords = np.zeros((3, 200), np.float32)
        i = 0
        self.points = PointCollection(mode="raw", transform=self.transform, viewport=self.viewport, color="local")
        # for seq in dataset._provideSequences():
        #     coords[0, i] = i / 10.0
        #     coords[1, i] = i / 10.0 - 5
        #     coords[2, i] = i / 10.0 * 3
        #     i += 1

        values = np.column_stack([input[:, 0], input[:, 1], target[:, 0]])
        colors = np.column_stack([target[:, 0] * 4,
                                  np.zeros(values.shape[0]),
                                  target[:, 1] * 4,
                                  # np.zeros([values.shape[0], 2]),
                                  np.ones(values.shape[0]) * 0.7])
        # color=np.random.uniform(0.0, 1, (p.shape[0], 4))
        self.points.append(values, color=colors)

    # def displayOutputsByAllWeights(self):
    #     coords = np.array(self.neuralNet.buildOutputsByAllWeights(self.diagramOptions))
    #     print("Min %f" % coords[:, :, 2].min())
    #     minInds = coords[:, :, 2].argmin(axis=1)
    #     # print(minInds)
    #     self.lastAllWeightsResults = coords
    #
    #     color = np.array([0.5, 0.1, 1, 0.7])
    #     # reshape(colors, color, coords.shape)
    #     colors = np.zeros((coords.shape[0] * coords.shape[1], 4)) + color
    #     # np.broadcast(colors, color, coords.shape)
    #     for i1 in range(coords.shape[0]):
    #         colors[i1 * coords.shape[1] + minInds[i1]] = (1, 0.1, 0.1, 0.7)
    #
    #     coords = reshape(coords, (coords.shape[0] * coords.shape[1], coords.shape[2]))
    #     coords[:, 2] *= 5
    #     self.points = PointCollection(mode="raw", transform=self.transform, viewport=self.viewport, color="local")
    #     self.points.append(coords, color=colors)
    #     self.resetTransform(2)

    def displayOutputsDiagram(self):
        coords = np.zeros((self.diagramOptions.stepCount * self.diagramOptions.stepCount, 3),
                          np.double)
        self.neuralNet.buildOutputsDiagram(coords, self.diagramOptions)
        # print(coords)

        minValue = coords[:, 2].min()
        maxValue = coords[:, 2].max()
        print("Conns %d, %d: min %f, max %f" % \
              (self.diagramOptions.xDimInd, self.diagramOptions.yDimInd, minValue, maxValue))
        if abs(minValue) > 0.5:
            coords[:, 2] -= minValue
            maxValue -= minValue
        if (maxValue - minValue < 0.3):
            coords[:, 2] /= max(maxValue - minValue, maxValue, 0.1)
            minValue /= max(maxValue - minValue, maxValue, 0.1)

        # vs = [self.transform.theta, self.transform.phi, self.transform.zoom]
        # self.points2 = self.points
        # self.points2.append(coords, color=(0, 0, 1, 0.5))
        self.points = PointCollection(mode="raw", transform=self.transform, viewport=self.viewport, color="local")
        self.points.append(coords, color=(0.5, 0, 1, 0.7))
        self.resetTransform()
        # self.transform.zoom = vs[2] + 0.00001
        # self.transform.theta = vs[0]
        # self.transform.phi = vs[1]
        # self.window.attach(self.transform)
        # self.window.attach(self.viewport)

    def displayAllWeightsDiagrams(self):
        diagramCount = self.neuralNet.getParamCount() / 2
        pairedValueCount = self.diagramOptions.stepCount * self.diagramOptions.stepCount * \
                diagramCount * 2
        coords = np.zeros((pairedValueCount, 3), np.double)
        valueCount = self.neuralNet.buildAllWeightsDiagrams(coords, self.diagramOptions)
        # print(coords)
        if valueCount != coords.shape[0]:
            coords = coords[:valueCount, :]

        minValue = coords[:, 2].min()
        maxValue = coords[:, 2].max()
        print("Min %f, max %f" % \
              (minValue, maxValue))
        if valueCount >= pairedValueCount:
            coordsHalf2 = coords[pairedValueCount / 2 :, 2]
            print("Test min %f, max %f" % \
                  (coordsHalf2.min(), coordsHalf2.max()))

        if abs(minValue) > 0.5:
            coords[:, 2] -= minValue
            maxValue -= minValue
        if (maxValue - minValue < 0.3):
            coords[:, 2] /= max(maxValue - minValue, maxValue, 0.1)
            minValue /= max(maxValue - minValue, maxValue, 0.1)

        coords[0, 0] -= self.diagramOptions.step
        coords[1, 0] -= self.diagramOptions.step
        coords /= 20
        # self.points = PointCollection(mode="raw", transform=self.transform, viewport=self.viewport, color="local")
        # for i in range(self.points.__len__() - 1, -1, -1):
        self.points.clear()
        if valueCount < pairedValueCount:
            self.points.append(coords, color=(0.5, 0, 1, 0.7))
        else:
            self.points.append(coords[: pairedValueCount / 2 - 1, :], color=(0.5, 0, 0.8, 0.5))
            self.points.append(coords[pairedValueCount / 2 :, :], color=(0.8, 0, 0.4, 0.5))
        # self.points._update()
        # self.resetTransform()

    def onParamChanged(self, paramInd, value):
        self.neuralNet.setParamValue(paramInd, value)
        self.diagramDisplayMethod()

    def changeBestWeight(self):
        if self.lastAllWeightsResults is None:
            return
        self.neuralNet.changeBestWeight(self.lastAllWeightsResults, self.diagramOptions)

    def changeTrainSetSize(self, mult):
        newTrainSetSize = self.neuralNet.getTrainSetSize() * mult
        print("Setting train size to " + str(newTrainSetSize))
        self.neuralNet.setTrainSetSize(int(newTrainSetSize))

    def onTrainingFinished(self):
        print("Training finished")

    def onNewTrainingData(self, learnResult, trainIterNum):
        # t0 = datetime.datetime.now()
        pass


    def tryMultipleStarts(self):
        paramCount = self.neuralNet.getParamCount()
        bestResult = 3
        bestTestResult = 3
        for i in range(1000):
            params = np.random.random((paramCount))
            print("%d. Starting from parameters %s" % (i, params))
            for j in range(paramCount):
                self.neuralNet.setParamValue(j, params[j])
            self.neuralNet.doLearning(10)

            result = self.neuralNet.getSqError()
            testResult = self.neuralNet.getTestSqError()
            print("Result: %f, test %f" % (result , testResult))
            if bestResult > result:
                bestResult = result
                print("Best result")
            if bestTestResult > testResult:
                bestTestResult = testResult
                print("Best test result")

            self.netChecker.checkNetResults(True)


    def on_draw(self, dt):
        if self.frameNumber < 5:
            self.window.set_position(75, -5)  # Doesn't work in init nor in first on_draw call
            self.frameNumber += 1

        self.window.clear()
        self.ticks.draw()
        self.points.draw()
        # paths.draw()

    def on_key_press(self, key, modifiers):
        if key == ord('1'):
            self.resetTransform(1)
        if key == ord('2'):
            self.resetTransform(2)
        if key == ord('9'):
            for i in range(100):
                self.changeBestWeight()
                self.showCurParameters()
                self.displayOutputsByAllWeights()
            print("Weights: %s" % self.neuralNet.net.params)
        elif key == ord('A'):
            self.displayOutputsByAllWeights()
        elif key == ord('D'):
            self.displayOutputsDiagram()
        elif key == ord('T'):
            if modifiers == 0:
                self.neuralNet.doLearning(10)
            elif modifiers == keys.MOD_CTRL:
                self.neuralNet.doLearning(2500)
            self.showCurParameters()
            self.diagramDisplayMethod()
            # self.displayOutputsByAllWeights()
        elif key == ord('M'):
            self.changeBestWeight()
            self.showCurParameters()
            self.displayOutputsByAllWeights()

        elif key == keys.SPACE:
            self.diagramDisplayMethod()
        elif key == keys.HOME:
            self.diagramOptions.xDimInd = 0
            self.diagramOptions.yDimInd = 1
            self.diagramDisplayMethod()
        elif key == keys.END:
            self.diagramOptions.xDimInd = self.neuralNet.getParamCount() - 2
            self.diagramOptions.yDimInd = self.neuralNet.getParamCount() - 1
            self.diagramDisplayMethod()
        elif key == keys.PAGEDOWN:
            if modifiers == 0:
                if self.diagramOptions.xDimInd >= 2:
                    self.diagramOptions.xDimInd -= 2
                    self.diagramOptions.yDimInd -= 2
            elif modifiers == keys.MOD_CTRL:
                self.diagramOptions.step /= 5
            elif modifiers == keys.MOD_CTRL + keys.MOD_SHIFT:
                self.diagramOptions.step /= 2
            self.diagramDisplayMethod()
        elif key == keys.PAGEUP:
            if modifiers == 0:
                if self.diagramOptions.yDimInd + 2 < self.neuralNet.getParamCount():
                    self.diagramOptions.xDimInd += 2
                    self.diagramOptions.yDimInd += 2
            elif modifiers == keys.MOD_CTRL:
                self.diagramOptions.step *= 5
            elif modifiers == keys.MOD_CTRL + keys.MOD_SHIFT:
                self.diagramOptions.step *= 2
            self.diagramDisplayMethod()

        elif key == keys.NUM_ADD or key == keys.PLUS:
            self.changeTrainSetSize(2)
        elif key == keys.NUM_SUBTRACT or key == keys.MINUS:
            self.changeTrainSetSize(0.5)

    def resetTransform(self, transformNum = 2):
        # self.window.clear
        # return

        if transformNum == 1:
            self.transform.theta = 90
            self.transform.phi = 0
        elif transformNum == 2:
            self.transform.theta = 70
            self.transform.phi = 270
        self.transform.zoom = 10 * self.diagramOptions.step


import signal

app = None

def sigint_handler(*args):
    global app

    """Handler for the SIGINT signal."""
    sys.stderr.write('Exiting\n')
    app.exiting = True
    QtGui.QApplication.quit()

if __name__ == "__main__":

    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    # List ALL tensors.
    # print_tensors_in_checkpoint_file(file_name='State/checkpoint', tensor_name='', all_tensors=True)


    # app.use('pyside')
    app = QtGui.QApplication(sys.argv)

    signal.signal(signal.SIGINT, sigint_handler)
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    # window = app.Window(width=1000, height=760, color=(1, 1, 1, 0.5))
    mainWindow = QtMainWindow()
    mainWindow.show()
    setProcessPriorityLow()
    if 1:
        mainWindow.init()
        mainWindow.loadState()
        # mainWindow.onDisplayIntermResultsPressed()
        # mainWindow.onDisplayPressed()
        # mainWindow.onShowActivationsPressed()
        # mainWindow.onShowMultActTopsPressed()
        # mainWindow.onShowSortedChanActivationsPressed()
        # mainWindow.onSpinBoxValueChanged()
    else:
        mainWindow.fastInit()
    # mainWindow.paintRect()
    sys.exit(app.exec_())
    # mainWindow.saveState()
