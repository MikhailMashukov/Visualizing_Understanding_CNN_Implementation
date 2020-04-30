""" Result notes:
Reinitialization of neirons works bad or doesn't give serious effect.
It's bad that the resting neirons setups theirs surroundings to theirs old weights.
Maybe it's not so simple that some neirons are more useful than others. Maybe each (or some) neiron
works in conjunction with some others, which gives theirs final good results. And this looses at
reinitializations.
After reinitialization gradients of different neirons often becomes correlated (10_Mlab, 12_Mlab\Gradients...png).

With separated towers and full reinitialization or keeping of entire tower reinitializations
seems to work better - good results restores very quickly and reaches source results more reliably.
It looks as a good idea to reinitialize with small weights and to multiply the kept weights
in order to compensate decreasing of total outputs. But I tried this only with 8 towers and later
15_2_32Towers_BatchNorm_NoDropout_RatherQuicklyTrain1.5e-4_Test1.2e-3:
  more and more conv_2 filters got stucked on the same favorite images
16_4_Train8e-4_Test0.95e-3: maybe with further training test results improve even better,
  (with the initial learning rate) as for 16_5.
21_Correlations\21_3_BasedOn20_4_3 (8 towers): one tower correlates well with the 0 one, others - much less

Activity regularization doesn't seem to give better generalization. Training becomes much slower,
even with big networks. Number of active neirons really becomes smaller. Default coefficients l1/l2 0.01
simply kill training, values should be like 1e-6 for both l1 and l2. They maybe depend on previous layers
(if they also contain such regularization, the choice is smaller... maybe no, since each neiron
anyway generates a potentially valuable result).

AlexNet experiments:
VKI\6: 24 classes, 12400 train images, 4096 neirons at dense levels. Learned quickly, loss function didn't lower much
  until epoch 48 10000 images each, then in about 30 epochs lowered maybe 30 times. Seems overtrained on upper levels -
  bad test predictions, but best activations on lower levels looks like they detect features
10_2: most probably bad variance on one tower on conv_12 because relu beats tanh.
10_4: relu beats sigmoid on conv_13
10_5: different initial weights distribution amplitudes on different towers
16_2: tried applying of the same convolutions to shifted image in order to get different picture
  according to strides on the first 2 layers. Seems to give additional 3% on validataion data
16_3: tried simply stride 1 instead of shifting. The improvement seems the same as in 16_2

1000-classes ImageNet experiments:
18_5-18_11: high epsilon killed learning
18_13: people use warming up for several epochs, but for me it worsened variance a lot after 160 mini-epochs,
  further training only killed neirons finally
PyT5_1-5_2: much worser result with shorter warmup, but higher variance at all layers except dense_3

Further ideas.
  Visualization:
    * to investigate how weights of particular neirons changed during training;
    - to take for each neiron its source convolutions with theirs weights and to display;
    * to find neirons which are active for particular classes (always > some minimum and high average);
    * to calculate statistics for each neiron how it is activated at correct and at wrong predictions;
    * to average values for each channel, then multiply channels by them and to sum -
      there will be pixels importance map https://www.youtube.com/watch?v=SOEPNYu6Yzc, near 4:14:00;
    * to look at activity maps (importance map) on incorrectly classified images,
      maybe they are on incorrect features;
      https://raghakot.github.io/keras-vis/visualizations/class_activation_maps/;
    * to exclude incorrect (or all) of them during training from scratch;
    +- to shift/resize a bit input images and to use this for top activations calculation;
    * to take activations for two objects, to build path between them and to generate
      what corresponds to them;
    * to look at backprop.-generated max activation images during training. Whether they are more noisy
      at start? Whether regularizers make them less noisy?

  Training:
    - to implement division of neirons: each is divided onto two with close weights
      and theirs output connections get about 1/2 of initial strength;
    * to turn SE blocks initially off;

  Net architecture:
    * to add 1 + for passing source channels coefficients through SE bottleneck
      (maybe wouldn't make sense since bias can make the same);
    * to train a usual each-to-each or towers conv. network, then estimate dissimilarity
      of the obtained conv_1 filters, and divide most different onto horizontal and vertical groups
      in a matrix network;
    * to try max pooling with strides (1, 1) and 3 * 2 with strides (2, 1);
    -+ * to try 3D max pooling for neighbour channels;
    * to add towers to teached net and to make much higher noise to the teached part;
    * it's possible to implement convolution of only neighbour channels by stacking channels[:-10], channels[1:-9], ...
    * to apply the same convolutions to neighbour layers - to recognize the same on different scales;
    * to combine convolutions' weights with multiplication in matrix net;

  Augmentation:
    * to add blocks of noise to the source images;
    * to add "augmentation" into the middle of the network - shifting and so on on channels
    - to add batch normalization at the end of SE blocks;
    * to multiply weights when activations dimish;
    - to run training steps several times and to select best result;
    - to try to scale images for TF AlexNet;
    * to add regularizations to ResNet;
    * to make average pooling to 2 * 2;
"""

# import copy
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

try:
    import PyQt4.Qt
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtGui import QMainWindow
except:
    import PyQt5.Qt
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QMainWindow
# from PySide import QtGui
# import re
    # from glumpy import app, glm   # In case of "GLFW not found" copy glfw3.dll to C:\Python27\Lib\site-packages\PyQt4\
    # from glumpy.transforms import Position, Trackball, Viewport
    # from glumpy.api.matplotlib import *
    # import glumpy.app.window.key as keys
import os
# import random
# import sys
# import time
import threading
try:
    from scipy.misc import imsave
except:
    from imageio import imsave  # For scipy >= 1.2, but imresize needs additional searching

# sys.path.append(r"../Qt_TradeSim")
import AlexNetVisWrapper
import ImageNetsVisWrappers
import MnistNetVisWrapper
import MultActTops
from MyUtils import *
from VisUtils import *
# from ControlWindow import *
# from CppNetChecker import *
# from NeuralNet import *
# from PersistDiagram import *
# from PriceHistory import *

# Returns two groups most distant points count / 2 each
# among N points in N * 2 coords array
def getMostDistantPoints(coords, count):
    # Converting to polar coordinates
    center = np.mean(coords, axis=0)
    deltas = coords - center[np.newaxis, :]
    dists = np.sqrt(np.square(deltas[:, 0]) + np.square(deltas[:, 1]))       # R
    angles = np.arctan(deltas[:, 1] / deltas[: , 0])                         # Theta
    angles[deltas[:, 0] < 0] += math.pi

    anglePartCount = 16
    angleParts = np.array(np.floor((angles / math.pi + 0.5) / 2 * anglePartCount), dtype=int) % anglePartCount
    assert angleParts.min() >= 0 and angleParts.max() < anglePartCount
    countByAngles = [0] * anglePartCount
    sortedByDistInds = dists.argsort()
    moreDistantInds = sortedByDistInds[coords.shape[0] // 2 : ]
    for i in moreDistantInds:
        countByAngles[angleParts[i]] += 1

    maxEst = 0
    for i in range(anglePartCount // 2):
        curEst = countByAngles[i] + countByAngles[i + anglePartCount // 2] + \
                min(countByAngles[i], countByAngles[i + anglePartCount // 2])
        if maxEst < curEst:
            maxEst = curEst
            selectedAngleParts = [i, i + anglePartCount // 2]
            # selectedAngle = (i / float(anglePartCount) - 0.5) * math.pi

    groups = [[], []]
    groupSize = count // 2
    anglePartDelta = 1
    while len(groups[0]) < groupSize or len(groups[1]) < groupSize:
        # if anglePartDelta == 0:
        #     curAngleParts = selectedAngleParts
        # else:
        curAngleParts = [[(selectedAngleParts[0] + anglePartDelta) % anglePartCount,
                              (selectedAngleParts[0] - anglePartDelta + anglePartCount) % anglePartCount],
                             [(selectedAngleParts[1] + anglePartDelta) % anglePartCount,
                              (selectedAngleParts[1] - anglePartDelta + anglePartCount) % anglePartCount]]
        if anglePartDelta == 1:
            curAngleParts[0].append(selectedAngleParts[0])
            curAngleParts[1].append(selectedAngleParts[1])
        for i in reversed(moreDistantInds):
            if angleParts[i] in curAngleParts[0]:
                if len(groups[0]) < groupSize:
                    groups[0].append(i)
                    # groups[0].append(coords[i])
            elif angleParts[i] in curAngleParts[1]:
                if len(groups[1]) < groupSize:
                    groups[1].append(i)
                    # groups[1].append(coords[i])
        print('Angles %s: %d' % (str(curAngleParts[0]), len(groups[0])))
        print('Angles 2 %s: %d' % (str(curAngleParts[1]), len(groups[1])))
        anglePartDelta += 1

    groups.append(moreDistantInds[-groupSize : ])
    groups.append(moreDistantInds[-groupSize * 3 : -groupSize])
    return groups


class QtMainWindow(QMainWindow): # , DeepMain.MainWrapper):
    c_channelMargin = 2
    c_channelMargin_Top = 5
    AllEpochs = -2

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        # super(QtMainWindow, self).__init__(parent)
        # DeepMain.MainWrapper.__init__(self, DeepOptions.studyType)
        self.exiting = False
        self.cancelling = False
        self.lastAction = None
        self.lastActionStartTime = None
        self.netWrapper = AlexNetVisWrapper.CAlexNetVisWrapper()
        # self.netWrapper = ImageNetsVisWrappers.CImageNetVisWrapper()
        # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper()
        # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper3_Towers()
        # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper4_Matrix()
        # self.netWrapper = MnistNetVisWrapper.CMnistVisWrapper5_DeeperTowers()
        self.activationCache = self.netWrapper.activationCache
        self.imageDataset = self.netWrapper.getImageDataset()
        self.tensorFlowLock = threading.Lock()
        # self.savedNetEpochs = None
        # self.curEpochNum = 0
        self.weightsBeforeReinit = None
        self.weightsReinitInds = None
        self.weightsReinitEpochNum = None
        self.initUI()
        # self.initAlexNetUI()

        # self.showControlWindow()

        self.iterNumLabel.setText('Epoch 0')
        self.maxAnalyzeChanCount = 70

    def init(self):
        # DeepMain.MainWrapper.__init__(self, DeepOptions.studyType)
        # DeepMain.MainWrapper.init(self)
        # DeepMain.MainWrapper.startNeuralNetTraining()
        # self.net = self.netTrader.net
        # self.fastInit()
        self.mousePressPos = None

        if not os.path.exists('Data/BestActs'):
            os.makedirs('Data/BestActs')
        if not os.path.exists('Data/Correlations'):
            os.makedirs('Data/Correlations')
        # os.makedirs('Data/Weights')
        self.loadNetStateList()
        self.epochComboBox.setCurrentIndex(self.epochComboBox.count() - 1)

        if 0:   #d_
            t0 = time.time()
            for i in range(10240):
                if i > 0 and i % 256 == 0:
                    print('%d images read: %.2f s' % (i, time.time() - t0))
                self.imageDataset.getImage(i, 'source')

    def initUI(self):
        self.setGeometry(100, 40, 1100, 700)
        self.setWindowTitle(self.getTitleForWindow())

        c_buttonWidth = 50
        c_minButtonWidth = 40
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
        self.iterNumLabel.setMinimumWidth(250)
        # self.iterNumLabel.setGeometry(x, y, 200, c_buttonHeight)
        curHorizWidget.addWidget(self.iterNumLabel)

        self.epochComboBox = QtGui.QComboBox(self)
        self.epochComboBox.currentIndexChanged.connect(self.onEpochComboBoxChanged)
        # self.epochComboBox.currentIndexChanged.connect(self.onSpinBoxValueChanged)
        self.iterNumLabel.setMinimumWidth(150)
        curHorizWidget.addWidget(self.epochComboBox)

        button = QtGui.QPushButton('Load state', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(lambda: self.onLoadStatePressed())
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Save state', self)  # and/or caches
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(lambda: self.saveState())
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Save act.', self)
        button.clicked.connect(self.saveActsForCorrelations)
        curHorizWidget.addWidget(button)

        # # x += c_buttonWidth + c_margin
        # button = QtGui.QPushButton('1 epoch', self)
        # button.clicked.connect(lambda: self.onDoItersPressed(100))
        # curHorizWidget.addWidget(button)

        lineEdit = QtGui.QLineEdit(self)
        # lineEdit.setValidator(QtGui.QIntValidator(1, 999999))
        lineEdit.setText(str(self.netWrapper.getRecommendedLearnRate()))
        lineEdit.setMaximumWidth(100)
        curHorizWidget.addWidget(lineEdit)
        self.learnRateEdit = lineEdit

        spinBox = QtGui.QSpinBox(self)
        spinBox.setRange(1, 1000000)
        spinBox.setValue(150000)
        spinBox.setSingleStep(100)
        spinBox.setMaximumWidth(120)
        curHorizWidget.addWidget(spinBox)
        self.iterCountEdit = spinBox

        button = QtGui.QPushButton('I&terations', self)
        button.setMinimumWidth(c_minButtonWidth)
        # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        button.clicked.connect(lambda: self.onDoItersPressed(int(self.iterCountEdit.text())))
        button.setToolTip('Run learning iterations')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Spec. iter.', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(lambda: self.onSpecialDoItersPressed(int(self.iterCountEdit.text())))
        button.setToolTip('Run iterations on selected samples')
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('+ learn. rate', self)
        # button.clicked.connect(lambda: self.onIncreaseLearningRatePressed())
        # curHorizWidget.addWidget(button)
        #
        button = QtGui.QPushButton('Reinit. worst n.', self)
        button.clicked.connect(lambda: self.onReinitWorstNeironsPressed())
        button.setToolTip('Reinitialize worst neirons')
        curHorizWidget.addWidget(button)
        layout.addLayout(curHorizWidget)

        # Widgets line 2 - "Mouse move..."
        y += c_buttonHeight + c_margin
        x = c_margin
        curHorizWidget = QtGui.QHBoxLayout()
        self.infoLabel = QtGui.QLabel(self)
        # self.infoLabel.setGeometry(x, y, self.width() - c_margin * 2, c_buttonHeight)
        # self.infoLabel.setGeometry(x, y, 550, c_buttonHeight * 3)
        self.infoLabel.setMinimumWidth(350)
        # controlsRestrictorWidget = QtGui.QWidget();  # Example how to restrict width, but it doesn't work, parent already set
        # controlsRestrictorWidget.setLayout(curHorizWidget);
        # controlsRestrictorWidget.setMaximumWidth(800);
        curHorizWidget.addWidget(self.infoLabel)

        # # x += 200 + c_margin
        # self.datasetComboBox = QtGui.QComboBox(self)
        # # self.datasetComboBox.setGeometry(x, y, 200, c_buttonHeight)
        # self.datasetComboBox.setMaximumWidth(60)
        # self.datasetComboBox.currentIndexChanged.connect(self.onDatasetChanged)
        # curHorizWidget.addWidget(self.datasetComboBox)

        x += 150 + c_margin
        self.blockComboBox = QtGui.QComboBox(self)
        # self.blockComboBox.setGeometry(x, y, 300, c_buttonHeight)
        self.blockComboBox.setMaximumWidth(100)
        # frame = QtGui.QFrame()
        # frame.add
        self.blockComboBox.currentIndexChanged.connect(self.onBlockChanged)
        curHorizWidget.addWidget(self.blockComboBox)

        spinBox = QtGui.QSpinBox(self)
        spinBox.setMaximumWidth(100)
        spinBox.setRange(0, int(2e9))
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

        button = QtGui.QPushButton('Correlat. an.', self)
        button.clicked.connect(self.onCorrelationAnalyzisPressed)
        button.setToolTip('Performs current layer\'s activations correlation analyzis among towers')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Correlat. to model 2', self)
        button.clicked.connect(self.onCorrelationToOtherModelPressed)
        button.setToolTip('Performs current layer\'s activations correlation analyzis to towers of another model')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Set towers weights', self)
        button.clicked.connect(self.onSetTowerWeightsPressed)
        curHorizWidget.addWidget(button)
        layout.addLayout(curHorizWidget)

        # Widgets line 3
        y += c_buttonHeight + c_margin
        x = c_margin
        curHorizWidget = QtGui.QHBoxLayout()
        # self.style().toolTipTimeouts[curHorizWidget] = 0      # 'QCommonStyle' object has no attribute 'toolTipTimeouts'
        # button = QtGui.QPushButton('test', self)
        # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        # button.clicked.connect(lambda: self.onTestPress())
        # curHorizWidget.addWidget(button)

        # lineEdit = QtGui.QLineEdit(self)
        # lineEdit.setValidator(QtGui.QIntValidator(1, 999999))
        # lineEdit.setText("1")
        # curHorizWidget.addWidget(lineEdit)
        spinBox = QtGui.QSpinBox(self)
        spinBox.setRange(1, 999999)
        spinBox.setValue(200)
        spinBox.valueChanged.connect(lambda: self.onSpinBoxValueChanged())
        curHorizWidget.addWidget(spinBox)
        self.imageNumEdit = spinBox

        button = QtGui.QPushButton('Show &image', self)
        # button.setGeometry(x, y, c_buttonWidth, c_buttonHeight)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowImagePressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Show &act.', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowActivationsPressed)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Act. est. by images', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowActEstByImagesPressed)
        button.setToolTip('Show activations estimations, one column per image')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Show &weights', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.showWeightsImage2)
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('t-SNE', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowImagesWithTSnePressed)
        button.setToolTip('Show images, distributed in accordance with activations t-SNE')
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('Show act. tops', self)     # Doesn't give interesting results
        # button.setMinimumWidth(c_minButtonWidth)
        # button.clicked.connect(self.onShowActTopsPressed)
        # curHorizWidget.addWidget(button)

        self.multActTopsButtonText = 'My &mult. act. tops'
        button = QtGui.QPushButton(self.multActTopsButtonText, self)
        button.setMinimumWidth(c_minButtonWidth)
        self.multActTopsButton = button
        button.clicked.connect(self.onShowMultActTopsPressed)
        button.setToolTip('Show my activations tops by multiple images')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Multith. act. tops', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.calcMultActTops_MultiThreaded)
        button.setToolTip('Multithreaded preparation of activations tops for all epochs')
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('Show all images tops', self)
        # button.setMinimumWidth(c_minButtonWidth)
        # button.clicked.connect(self.onShowActTopsFromCsvPressed)
        # curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Average &grad.', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowGradientsPressed)
        button.setToolTip('Show average (by specified number of first images) gradients')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Grad. by images', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onGradientsByImagesPressed)
        button.setToolTip('Show gradients by images, for one image at vertical line')
        curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('Worst images', self)
        button.setMinimumWidth(c_minButtonWidth)
        button.clicked.connect(self.onShowWorstImagesPressed)
        button.setToolTip('Show images with maximal prediction errors')
        curHorizWidget.addWidget(button)

        # button = QtGui.QPushButton('Pred. history', self)
        # button.setMinimumWidth(c_minButtonWidth)
        # button.clicked.connect(self.onShowPredictionHistoryPressed)
        # # button.setToolTip('')
        # curHorizWidget.addWidget(button)

        button = QtGui.QPushButton('&Cancel', self)
        button.setMinimumWidth(c_minButtonWidth)
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
                                                 #    hspace=0.3)

        # self.blockComboBox.setCurrentIndex(9)  # max_pool_22
        self.blockComboBox.setCurrentIndex(7)
        # self.lastAction = self.onShowActTopsPressed   #d_

    def getTitleForWindow(self):
        path = os.getcwd()
        try:
            pathBlocks = os.path.split(path)
            pathBlocks2 = os.path.split(pathBlocks[0])
            path = '%s/%s' % (pathBlocks2[1], pathBlocks[1])
        except:
            pass

        return 'Visualization %s %d' % (path, os.getpid())

    def showProgress(self, str, processEvents=True):
        print(str)
        # self.setWindowTitle(str)
        self.infoLabel.setText(str)
        if processEvents:
            PyQt4.Qt.QCoreApplication.processEvents()

    def startAction(self, actionFunc):
        self.lastAction = actionFunc
        self.lastActionStartTime = datetime.datetime.now()
        self.cancelling = False

    # TODO def startVisualizationAction(self, actionFunc):
    #     self.lastAction = actionFunc
    #     self.lastActionStartTime = datetime.datetime.now()
    #     self.cancelling = False

    def getSelectedEpochNum(self):
        if self.netWrapper.isLearning:
            return -5         # Ok to take the current epoch, tracked not by this object, but by the net wrapper
        epochNum = -6
        epochText = self.epochComboBox.currentText().lower()
        if epochText.find('all') >= 0:
            return self.AllEpochs
        pos = epochText.find('epoch')
        if pos >= 0:
            epochNum = int(epochText[pos + len('epoch') : ])
        return epochNum

    def getPrevLayerName(self, layerName):
        import re

        result = re.search(r'conv_(\d+)(.*)', layerName)
        if result:
            layerNum = int(result.group(1))
            return 'conv_%d%s' % (layerNum - 1, result.group(2))
        else:
            raise Exception('Can\'t decode layer number')

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

    def getChannelToAnalyzeCount(self, data):
        chanCount = data.shape[0]
        if self.maxAnalyzeChanCount and chanCount > self.maxAnalyzeChanCount:
            chanCount = self.maxAnalyzeChanCount
        colCount = math.ceil(math.sqrt(chanCount) * 1.15 / 2) * 2
        if colCount in [4, 6] and chanCount % (colCount - 1) == 0:
            colCount -= 1
        chanCount = chanCount // colCount * colCount
        return (chanCount, colCount)


    def closeEvent(self, event):
        self.exiting = True
        self.netWrapper.cancelling = True
        print("Close event")

    def onCancelPressed(self):
        # self.lastAction = None
        self.cancelling = True
        self.netWrapper.cancelling = True
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
        self.showImage(ax, imageData)
        # ax.imshow(imageData, extent=(-100, 127, -100, 127), aspect='equal')
        self.canvas.draw()

    def clearFigure(self):
        self.figure.clear()
        self.mainSubplotAxes = None
        self.figure.set_tight_layout(True)

    def getMainSubplot(self):
        if not hasattr(self, 'mainSubplotAxes') or self.mainSubplotAxes is None:
            self.mainSubplotAxes = self.figure.add_subplot(self.gridSpec[:, 1])
        return self.mainSubplotAxes

    def showImage(self, ax, imageData):
        # ax.clear()    # Including clear here is handy, but not obvious
        if len(imageData.shape) >= 3:
            if imageData.shape[2] > 1:
                return ax.imshow(imageData) # , aspect='equal')
            else:
                imageData = np.squeeze(imageData, axis=2)
                # if imageData.dtype == np.float32:
                #     imageData *= 255
                return ax.imshow(imageData, cmap='Greys_r')
        else:
            return ax.imshow(imageData, cmap='Greys_r')

    def drawFigure(self):
        self.canvas.draw()

    def onShowActivationsPressed(self):
        self.startAction(self.onShowActivationsPressed)
        epochNum = self.getSelectedEpochNum()
        imageNum = self.getSelectedImageNum()
        layerName = self.blockComboBox.currentText()

        if epochNum == self.AllEpochs:
            epochNums = self.netWrapper.getSavedNetEpochs()
            if len(epochNums) == 0:
                self.showProgress('Activations error: no saved epochs')
                epochNum = -10

        if epochNum == self.AllEpochs:
            dataList = []
            t0 = datetime.datetime.now()

            for curEpochNum in self.netWrapper.getSavedNetEpochs():
                dataList.append(self.netWrapper.getImagesActivations_Batch(layerName, [imageNum], curEpochNum)[0])
                t = datetime.datetime.now()
                if (t - t0).total_seconds() >= 1:
                    self.showProgress('Analyzed epoch %d' % curEpochNum)
                    t0 = t
                    if self.cancelling or self.exiting:
                        self.showProgress('Cancelled')
                        break
            activations = np.stack(dataList, axis=0)
            self.showProgress('Activations: %s, min %.4f, max %.4f (%s)' % \
                    (str(activations.shape), activations.min(), activations.max(),
                     str([int(v[0]) for v in np.where(activations == activations.max())])))

            drawMode = 'map'
            stdData = None
            if len(activations.shape) > 2:
                axisToStick = tuple(range(2, len(activations.shape)))
                stdData = np.std(activations, axis=axisToStick)
                activations = np.abs(activations)
                activations = np.mean(activations, axis=axisToStick)
        else:
            activations = self.netWrapper.getImageActivations(layerName, imageNum, epochNum)
            self.showProgress('Activations: %s, max %.4f (%s)' % \
                    (str(activations.shape), activations.max(),
                     str([int(v[0]) for v in np.where(activations == activations.max())])))

            drawMode = 'map'
            if len(activations.shape) == 2:   # Dense level scalars
                if activations.shape[1] < 50:
                    drawMode = 'plot'
                    activations = activations.flatten()
                else:
                    activations = np.reshape(activations, [activations.shape[1], 1, 1])
                    margin = 0
            else:
                activations = self.getChannelsToAnalyze(activations[0])
                margin = self.c_channelMargin

        if epochNum == self.AllEpochs:
            self.clearFigure()
            ax = self.getMainSubplot()

            im = ax.imshow(activations.transpose(), cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)

            if not stdData is None:
                ax = self.figure.add_subplot(self.gridSpec[1, 0])
                self.showImage(ax, stdData.transpose())
        else:
            self.figure.set_tight_layout(True)
            ax = self.getMainSubplot()
            ax.clear()

            # plt.subplots_adjust(left=0.01, right=data.shape[0], bottom=0.1, top=0.9)
            if drawMode == 'map':
                colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
                data = layoutLayersToOneImage(np.sqrt(activations),
                                              colCount, margin)
                ax.imshow(data, cmap='plasma')
            else:
                ax.plot(activations)
                if ax.get_ylim()[0] > ax.get_ylim()[1]:
                    ax.invert_yaxis()


        self.canvas.draw()

    def onShowActEstByImagesPressed(self):
        self.startAction(self.onShowActEstByImagesPressed)
        epochNum = self.getSelectedEpochNum()
        firstImageCount = max(100, self.getSelectedImageNum())
        layerName = self.blockComboBox.currentText()

        imagesActs = self.netWrapper.getImagesActivations_Batch(
                layerName, range(1, firstImageCount + 1), epochNum)
        self.showProgress('Activations: %s, max %.4f (%s)' % \
                (str(imagesActs.shape), imagesActs.max(),
                 str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
        ests = self.getEstimations(imagesActs)

        self.clearFigure()
        # ax = self.getMainSubplot()

        # ax = self.figure.add_subplot(self.gridSpec[0, 0])

        if 1:
            (ests, data2) = self.sortEstimations(ests, imagesActs)
        else:
            data2 = imagesActs
        data2 = data2.transpose([1, 0] + list(range(2, len(imagesActs.shape))))
        self.showGradients(ests.transpose([1, 0]), data2, False)
        # self.showGradients(ests, imagesActs, True)  # Transpose will be made inside, but also undesirable log10

    def onShowImagesWithTSnePressed(self):
        self.startAction(self.onShowImagesWithTSnePressed)
        epochNum = self.getSelectedEpochNum()
        firstImageCount = max(300, self.getSelectedImageNum())
        layerName = self.blockComboBox.currentText()

        imagesActs = self.netWrapper.getImagesActivations_Batch(
                layerName, range(1, firstImageCount + 1), epochNum)
        self.showProgress('Activations: %s, max %.4f (%s)' % \
                (str(imagesActs.shape), imagesActs.max(),
                 str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
        ests = self.getEstimations(imagesActs)

        if 1:
            self.showTSneByImages(ests, firstImageCount)
        else:
            self.showTSneByChannels(ests)

    def showTSneByImages(self, ests, firstImageCount):
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, verbose=9, perplexity=40, n_iter=250, random_state=1)
        # for iterCount in range(250, 2000000, 50):
        iterCount = 3000
        while iterCount < 6000000:
            tsne.n_iter = iterCount
            tsne.n_iter_without_progress = 250
            tsneCoords = tsne.fit_transform(ests)
            self.showProgress('T-SNE: %d iterations made' % iterCount)

            ax = self.getMainSubplot()
            ax.clear()
            (extent, imageSize) = self.getTsneResultsExtent(tsneCoords)
            selectedImageInds = getMostDistantPoints(tsneCoords, tsneCoords.shape[0] // 4)
            np.savetxt('Data/DistantImageNums1.txt', selectedImageInds[0], fmt='%d', delimiter='')
            np.savetxt('Data/DistantImageNums2.txt', selectedImageInds[1], fmt='%d', delimiter='')

            for i, coords in enumerate(tsneCoords):
                imageNum = i + 1
                imageData = self.imageDataset.getImage(imageNum, 'cropped')
                imageData = np.squeeze(imageData, axis=2)
                ax.imshow(imageData, cmap='Greys_r', alpha=0.05,
                          extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))

            for imageInd in selectedImageInds[0]:
                imageNum = imageInd + 1
                imageData = self.imageDataset.getImage(imageNum, 'cropped')
                imageData = np.squeeze(imageData, axis=2)
                coords = tsneCoords[imageInd]
                ax.imshow(imageData, cmap='Greys_r', alpha=0.35,
                          extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))
            for imageInd in selectedImageInds[1]:
                imageNum = imageInd + 1
                imageData = self.imageDataset.getImage(imageNum, 'cropped')
                imageData = np.squeeze(imageData, axis=2)
                coords = tsneCoords[imageInd]
                ax.imshow(imageData, cmap='plasma', alpha=0.35,
                          extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))

            ax.imshow(imageData, cmap='Greys_r', alpha=0,
                      extent=extent)

            ax = self.figure.add_subplot(self.gridSpec[0, 0])
            ax.clear()
            labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
            ax.scatter(tsneCoords[:, 0], tsneCoords[:, 1], c=labels, alpha=0.6, cmap='plasma')

            self.canvas.draw()
            self.showProgress('T-SNE: %d iterations displayed' % iterCount)

            if self.cancelling or self.exiting:
                return
            iterCount = int(iterCount * 1.2)
            break
            # iterCount += 50

    def showTSneByChannels(self, ests):
        from sklearn.manifold import TSNE

        self.needShowCurMultActTops = False
        calculator = self.fillMainMultActTopsOptions()
        calculator.chanCount = ests.shape[1]
        if not calculator.checkMultActTopsInCache():
            (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
        else:
            (bestSourceCoords, processedImageCount) = (None, None)
        chanImages = calculator.prepareMultActTopsImageData(bestSourceCoords, processedImageCount)
        chanImages = np.squeeze(chanImages, axis=3)
        # resultImage = calculator.buildMultActTopsImage(bestSourceCoords, processedImageCount)

        tsne = TSNE(n_components=2, verbose=9, perplexity=30, n_iter=250, random_state=1)
        # for iterCount in range(250, 2000000, 50):
        iterCount = 250
        assert len(ests.shape) == 2
        tsneData = np.transpose(ests)
        while iterCount < 1000000:
            tsne.n_iter = iterCount
            tsne.n_iter_without_progress = 250
            tsneCoords = tsne.fit_transform(tsneData)
            self.showProgress('T-SNE: %d iterations made' % iterCount)

            ax = self.getMainSubplot()
            ax.clear()
            (extent, imageSize) = self.getTsneResultsExtent(tsneCoords)
            for chanInd, coords in enumerate(tsneCoords):
                imageData = chanImages[chanInd]
                ax.imshow(imageData, cmap='Greys_r', alpha=0.5,
                          extent=(coords[0], coords[0] + imageSize, coords[1], coords[1] + imageSize))

            ax.imshow(imageData, cmap='Greys_r', alpha=0,
                      extent=extent)
                    # (tsneCoords[:, 0].min(), tsneCoords[:, 0].max() + imageSize,
                    #  tsneCoords[:, 1].min(), tsneCoords[:, 1].max() + imageSize))

            # ax = self.figure.add_subplot(self.gridSpec[0, 0])
            # ax.clear()
            # labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
            # ax.scatter(tsneCoords[:, 0], tsneCoords[:, 1], c=labels, alpha=0.6, cmap='plasma')

            self.canvas.draw()
            self.showProgress('T-SNE: %d iterations displayed' % iterCount)
            if self.cancelling or self.exiting:
                return
            # iterCount = int(iterCount * 1.25)
            iterCount += 50

    def getTsneResultsExtent(self, tsneCoords):
        c_extentSkipCount = 5
        c_maxEnlargeMult = 0.1

        # Skipping first most distant (min and max x and y) points
        # and then returning those of them which are not too far
        sortedXs = np.sort(tsneCoords[:, 0])
        sortedYs = np.sort(tsneCoords[:, 1])
        extentWidth = sortedXs[-c_extentSkipCount] - sortedXs[c_extentSkipCount]
        extentHeight = sortedYs[-c_extentSkipCount] - sortedYs[c_extentSkipCount]
        # extent = (sortedXs[c_extentSkipCount], sortedXs[-c_extentSkipCount],
        #           sortedYs[c_extentSkipCount], sortedYs[-c_extentSkipCount])
        imageSize = max(extentWidth, extentHeight) / \
                math.sqrt(tsneCoords.shape[0]) / 2
        extent = [sortedXs[c_extentSkipCount], sortedXs[-c_extentSkipCount],
                  sortedYs[c_extentSkipCount], sortedYs[-c_extentSkipCount]]
        for i in range(c_extentSkipCount - 1, -1, -1):
            if extent[0] - extentWidth * c_maxEnlargeMult <= sortedXs[i]:
                extent[0] = sortedXs[i]
            if extent[1] + extentWidth * c_maxEnlargeMult >= sortedXs[-i - 1]:
                extent[1] = sortedXs[-i - 1]
            if extent[2] - extentWidth * c_maxEnlargeMult <= sortedYs[i]:
                extent[2] = sortedYs[i]
            if extent[3] + extentWidth * c_maxEnlargeMult >= sortedYs[-i - 1]:
                extent[3] = sortedYs[-i - 1]
        extent[1] += imageSize
        extent[3] += imageSize
        # center = ((sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount]) / 2,
        #           (sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount]) / 2,
        # extent = ((sortedXs[-c_extentSkipCount] + sortedXs[c_extentSkipCount])
        return (extent, imageSize)

    def getEstimations(self, imagesActs):
        shape = imagesActs.shape
        if len(shape) == 2:
            return imagesActs
        else:
            data = abs(imagesActs)
            # counts = np.zeros(shape)
            # counts[data >= 0.1] = 1
            # counts = np.sum(counts, axis=tuple(range(2, len(shape))))
            mean = np.mean(data)
            # boundary = 0.2
            boundary = mean / 1.8
            counts = np.count_nonzero(data >= boundary, axis=tuple(range(2, len(shape))))

            means = np.mean(imagesActs, axis=tuple(range(2, len(shape))))
            return np.concatenate([counts, means], axis=1)

    def sortEstimations(self, ests, imagesActs):
        # First dimension here is images

        if 0:
            # Sorting by sum of estimations
            sum = np.sum(ests, axis=1)
            newInds = sum.argsort()
            ests = ests[newInds, :]
            # imagesActs = ests[np.sum(imagesActs, axis=1).argsort(), :]
            imagesActs = imagesActs[newInds, :]
        else:
            firstImageCount = ests.shape[0]
            labels = np.array(self.imageDataset.getNetSource()[1][:firstImageCount], dtype=float)
            mean = np.mean(ests, axis=1)
            mean -= mean.min()
            labels += mean / (mean.max() + 1e-15) / 2
            newInds = labels.argsort()
            ests = ests[newInds]
            imagesActs = imagesActs[newInds]

        return (ests, imagesActs)

    def onShowActTopsPressed(self):       # It would be desirable to move into MultActTops, but this requires time and the code is not necessary
        self.startAction(self.onShowActTopsPressed)
        imageNum = self.getSelectedImageNum()
        sourceImageData = self.imageDataset.getImage(imageNum, 'cropped')
        # alexNetImageData = self.imageDataset.getImage(imageNum, 'net')
        layerName = self.blockComboBox.currentText()
        activations = self.netWrapper.getImageActivations(layerName, imageNum)
        activations = self.getChannelsToAnalyze(activations[0])
        if len(activations.shape) == 1:
            return
            # activations = np.expand_dims(activations, axis=0)

        sourceBlockCalcFunc = self.netWrapper.get_source_block_calc_func(layerName)
        if sourceBlockCalcFunc is None:
            return
        colCount = math.ceil(math.sqrt(activations.shape[0]) * 1.15 / 2) * 2
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
        data = layoutLayersToOneImage(data, colCount, self.c_channelMargin)

        ax = self.figure.add_subplot(self.gridSpec[1, 0])
        ax.clear()
        self.showImage(ax, data)
        self.canvas.draw()

    def onShowActTopsFromCsvPressed(self):
        # Fast, based on data produced by activations.py
        self.startAction(self.onShowActTopsFromCsvPressed)
        calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
        options = calculator
        options.layerNum = self.blockComboBox.currentIndex() + 1
        calculator.showActTops_FromCsv()


    def fillMainMultActTopsOptions(self):
        calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
        options = calculator
        options.epochNum = self.getSelectedEpochNum()
        if options.epochNum is None:
            epochNums = self.netWrapper.getSavedNetEpochs()
            options.epochNum = epochNums[-1]
        layerName = self.blockComboBox.currentText()
        options.layerName = layerName
        options.embedImageNums = True
        options.imageToProcessCount = max(200 if self.netWrapper.name == 'mnist' else 20, \
                    self.getSelectedImageNum())

        # Getting activations to get channel count. They should be reread from cache later
        # try:
        activations1 = self.netWrapper.getImageActivations(
                            layerName, 1, options.epochNum)
        # except Exception as ex:
        #     self.showProgress("Error: %s. Looking for children layers" % str(ex))
        #     activations1 = self.net.getCombinedLayerImageActivations(layerName, 1, options.epochNum)
        activations1 = self.getChannelsToAnalyze(activations1[0])
        (options.chanCount, options.colCount) = self.getChannelToAnalyzeCount(activations1)

        calculator.progressCallback = QtMainWindow.TProgressIndicator(self, calculator)
        return calculator

    def onShowMultActTopsPressed(self):
        # My own implementation, from scratch, with images subblocks precision
        self.startAction(self.onShowMultActTopsPressed)
        calculator = self.fillMainMultActTopsOptions()
        if calculator.checkMultActTopsInCache():
            # No need to collect activations, everything will be taken from cache
            resultImage = calculator.showMultActTops(None)
            # calculator.saveMultActTopsImage(resultImage)
            return

        self.needShowCurMultActTops = False
        self.multActTopsButton.setText('Save current')
        try:
            self.multActTopsButton.clicked.disconnect()
        except e:
            pass
        self.multActTopsButton.clicked.connect(self.onShowCurMultActTopsPressed)

        # activations = self.getChannelsToAnalyze(self.netWrapper.getImageActivations(
        #           layerNum, 1, options.epochNum)[0])
        # print(activations)

        try:
            (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
        finally:
            self.multActTopsButton.setText(self.multActTopsButtonText)
            self.multActTopsButton.clicked.disconnect()
            self.multActTopsButton.clicked.connect(self.onShowMultActTopsPressed)

        if not self.exiting:
            resultImage = calculator.showMultActTops(bestSourceCoords, processedImageCount)
            calculator.saveMultActTopsImage(resultImage, processedImageCount)

    class TProgressIndicator:
        def __init__(self, mainWindow, calculator, threadInfo=None):
            self.mainWindow = mainWindow
            self.calculator = calculator
            self.threadInfo = threadInfo
            # self.mainWindowLock = threading.Lock()

        # Returns false if the process should be stopped.
        # Can show and save intermediate results. When called from showMultActTops this is not necessary
        # since they already were called and the image is about to be prepared, showed and/or saved
        def onMultActTopsProgress(self, infoStr, processedImageCount=None, curBestSourceCoords=None):
            if not self.threadInfo is None:
                infoStr = '%s: %s' % (self.threadInfo, infoStr)
           # with self.mainWindowLock:
            self.mainWindow.showProgress(infoStr)
                # Usually doesn't display anything from other threads

            if self.mainWindow.cancelling or self.mainWindow.exiting:
                return False
            elif self.mainWindow.needShowCurMultActTops and not curBestSourceCoords is None:
                self.mainWindow.needShowCurMultActTops = False
                resultImage = self.calculator.showMultActTops(curBestSourceCoords, processedImageCount)
                self.calculator.saveMultActTopsImage(resultImage, processedImageCount)
            return True

        def onEpochProcessed(self, infoStr):
            self.mainWindow.showProgress(infoStr)
            if self.mainWindow.cancelling or self.mainWindow.exiting:
                return False
            return True


    def calcMultActTops_MultiThreaded(self):
        self.startAction(self.calcMultActTops_MultiThreaded)
        calculator = self.fillMainMultActTopsOptions()
        self.needShowCurMultActTops = False
        epochNums = self.netWrapper.getSavedNetEpochs()
        calculator.epochNum = None
        calculator.threadCount = getCpuCoreCount() * 2
        mtCalculator = MultActTops.CMultiThreadedCalculator()

        # mtCalculator.run(calculator, epochNums)
        mtCalculator.run_MultiProcess(calculator, epochNums)

    def calcMultActTops_MultipleTfThreaded(self):
        self.startAction(self.calcMultActTops_MultiThreaded)
        calculator = self.fillMainMultActTopsOptions()
        epochNums = self.netWrapper.getSavedNetEpochs()
        calculator.epochNum = None

        import tensorflow as tf

        # All these graph tricks don't help - it throws exception from model.predict
        # "'_thread._local' object has no attribute 'value'.
        # And they consume a lot of memory for some reason (4 GB for 4 wrappers in total)
        self.netWrappers = []
        for _ in range(3):
            threadGraph = tf.Graph()

            with threadGraph.as_default():
                threadSession = tf.compat.v1.Session()  #  tf.Session()
                with threadSession.as_default():
                    netWrapper = MnistNetVisWrapper.CMnistVisWrapper( \
                           self.imageDataset, self.activationCache)
                    netWrapper._getNet(options.layerName)
                    netWrapper.threadGraph = threadGraph
                    netWrapper.threadSession = threadSession
                    # netWrapper.net.model._make_predict_function()
            # threadGraph.finalize()
            threadGraph.switch_to_thread_local()
            self.netWrappers.append(netWrapper)
        # tf.compat.v1.get_default_graph().finalize()

        options.threadCount = 3
        # threads = []
        calc_MultiThreaded(options.threadCount, self.calcMultActTops_MultipleTfThreadFunc,
                           epochNums, calculator)
        self.showProgress('%d threads finished' % options.threadCount)

    def calcMultActTops_MultipleTfThreadFunc(self, threadParams, options):
        downloadStats = threadParams.downloadStats
        self.showProgress('%s: started for %s%s' % \
              (threadParams.threadName, threadParams.ids[:20], \
               '...' if len(threadParams.ids) > 20 else ''))

        curThreadAdded = 0
        try:
            # t0 = datetime.datetime.now()
            # calculator = MultActTops.CMultActTopsCalculator(self, self.activationCache, self.netWrapper)
            epochNum = -1

            # import tensorflow as tf

            calculator = copy.copy(options)
            # with self.tensorFlowLock:
            #     g = tf.Graph()
            #     with g.as_default():
            #         calculator.netWrapper = MnistNetVisWrapper.CMnistVisWrapper(
            #                 self.imageDataset, self.activationCache)
            #         net = calculator.netWrapper._getNet()
            #         calculator.netWrapper.model._make_predict_function()
            #     g.finalize()
            #     print('%s: net inited' % threadParams.threadName)

            calculator.netWrapper = self.netWrappers[threadParams.ids[0] % 3]
            calculator.progressCallback = QtMainWindow.TProgressIndicator(
                    self, calculator, threadParams.threadName)
            with calculator.netWrapper.threadGraph.as_default():
                with calculator.netWrapper.threadSession.as_default():
                    for epochNum in threadParams.ids:
                        calculator.epochNum = epochNum
                        (bestSourceCoords, processedImageCount) = calculator.calcBestSourceCoords()
                        if self.cancelling or self.exiting:
                            break
                        resultImage = calculator.showMultActTops(bestSourceCoords, processedImageCount)
                        calculator.saveMultActTopsImage(resultImage)
                        self.showProgress('%s: epoch %d done' % (threadParams.threadName, epochNum))
        except Exception as errtxt:
            print('Exception at %s on epoch %d: %s' % (threadParams.threadName, epochNum, errtxt))

        with downloadStats.updateLock:
            downloadStats.finishedThreadCount += 1


    def onShowCurMultActTopsPressed(self):
        self.needShowCurMultActTops = True

    def onShowChanActivationsPressed(self):
        try:
            self.startAction(self.onShowChanActivationsPressed)
            epochNum = self.getSelectedEpochNum()
            imageNum = self.getSelectedImageNum()
            layerName = self.blockComboBox.currentText()
            chanNum = self.getSelectedChannelNum()
            activations = self.netWrapper.getImageActivations(layerName, imageNum, epochNum)
            activations = activations[0][chanNum]

            self.clearFigure()
            showCurLayerActivations = True
            if showCurLayerActivations:
                imageData = np.sqrt(activations)
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.imshow(imageData, cmap='plasma')

            vals = attachCoordinates(activations)        # Getting list of (x, y, value)
            sortedVals = vals[:, vals[2, :].argsort()]
            curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))

            # ax = self.figure.add_subplot(self.gridSpec[0, 0])
            # ax.clear()
            # ax.hist([1,2,3], bins=100)

            # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
            # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
            allWeights = self.netWrapper.getMultWeights(layerName)
            weights = allWeights[:, :, :, chanNum]
            if layerName[:6] == 'conv_1':
                prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
            else:
                prevLayerName = self.getPrevLayerName(layerName)
                prevLayerActivations = self.netWrapper.getImageActivations(prevLayerName, imageNum, epochNum)[0]
                        # Conv2 - 256 * 27 * 27
            self.showProgress('Prev. layer act.: %s, max %.4f (%s)' % \
                (str(prevLayerActivations.shape), prevLayerActivations.max(),
                 str([int(v[0]) for v in np.where(prevLayerActivations == prevLayerActivations.max())])))

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
            epochNum = self.getSelectedEpochNum()
            imageNum = self.getSelectedImageNum()
            layerNum = self.blockComboBox.currentIndex() + 1
            chanNum = self.getSelectedChannelNum()
            activations = self.netWrapper.getImageActivations(layerNum, imageNum, epochNum)
            activations = activations[0][chanNum]

            self.clearFigure()

            showCurLayerActivations = True
            if showCurLayerActivations:
                imageData = np.sqrt(activations)
                ax = self.figure.add_subplot(self.gridSpec[0, 0])
                ax.clear()
                ax.imshow(imageData, cmap='plasma')

            vals = attachCoordinates(activations)        # Getting list of (x, y, value)
            sortedVals = vals[:, vals[2, :].argsort()]
            curLayerMaxCoords = (int(sortedVals[0, -1]), int(sortedVals[1, -1]))

            # allWeights = self.getAlexNet().model.layers[13]._trainable_weights     # Conv3 - 3 * 3 * 256 * 384 and bias 384
            # allWeights = self.getAlexNet('conv_%d_weights' % layerNum)
            allWeights = self.netWrapper.getMultWeights('conv_%d' % layerNum)
            weights = allWeights[0].numpy()[:, :, :, chanNum]
            if layerNum == 1:
                prevLayerActivations = self.imageDataset.getImage(imageNum, 'cropped')
            else:
                prevLayerActivations = self.netWrapper.getImageActivations(layerNum - 1, imageNum, epochNum)[0]
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

            options = MultActTops.TMultActOpsOptions()
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

    # Another approach to visualize weights. Doesn't work well, especially with max pool. Because we sum
    # masks during theirs imposing to generate next layer's mask, but actually the same previous layer's
    # values are used
    def showWeightsImage2(self):
        try:
            self.startAction(self.showWeightsImage2)
            epochNum = self.getSelectedEpochNum()
            self.netWrapper.loadState(epochNum)
            layerName = self.blockComboBox.currentText()

            imageData = self.netWrapper.calcWeightsVisualization2(layerName)
            chanCount, colCount = self.getChannelToAnalyzeCount(imageData)
            imageData = layoutLayersToOneImage(imageData, colCount, 1, imageData.min())

            dirName = 'Data/WeightsVis/%s_%dChan' % \
                      (layerName, chanCount)
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            fileName = '%s/Weights_%s_Epoch%d.png' % \
                       (dirName, layerName, epochNum)
            imsave(fileName, imageData, format='png')
            print("Image saved to '%s'" % fileName)

            ax = self.getMainSubplot()
            ax.clear()
            ax.imshow(imageData)
            self.canvas.draw()
        except Exception as ex:
            self.showProgress("Error: %s" % str(ex))

    def onShowGradientsPressed(self):
        self.startAction(self.onShowGradientsPressed)
        epochNum = self.getSelectedEpochNum()
        firstImageCount = self.getSelectedImageNum()
        layerName = self.blockComboBox.currentText()

        data2 = None
        print("Getting data")
        if epochNum == self.AllEpochs:
            dataList = []
            t0 = datetime.datetime.now()
            for curEpochNum in self.netWrapper.getSavedNetEpochs():
                dataList.append(self.netWrapper.getGradients(layerName, 1, firstImageCount, curEpochNum, True))
                t = datetime.datetime.now()
                if (t - t0).total_seconds() >= 1:
                    self.showProgress('Analyzed epoch %d' % curEpochNum)
                    t0 = t
                    if self.cancelling or self.exiting:
                        self.showProgress('Cancelled')
                        break

            data = np.abs(np.stack(dataList, axis=0))
            if len(data.shape) == 5:
                data2 = np.mean(data, axis=(2))    # Averaging by other convolution channels dimension out of 2
            data = np.mean(data, axis=(1))
        else:
            data = np.abs(self.netWrapper.getGradients(layerName, 1, firstImageCount, epochNum, True))

        self.showProgress('Gradients: %s, max %.4f (%s)' % \
                (str(data.shape), data.max(), str([int(v[0]) for v in np.where(data == data.max())])))
        self.showGradients(data, data2, epochNum == self.AllEpochs)

    def onGradientsByImagesPressed(self):
        self.startAction(self.onGradientsByImagesPressed)
        epochNum = self.getSelectedEpochNum()
        firstImageCount = max(self.getSelectedImageNum(), 100)
        layerName = self.blockComboBox.currentText()

        data2 = None
        dataList = []
        t0 = datetime.datetime.now()
        for imageNum in range(1, firstImageCount + 1):
            dataList.append(self.netWrapper.getGradients(layerName, imageNum, 1, epochNum, True))
            t = datetime.datetime.now()
            if (t - t0).total_seconds() >= 1:
                self.showProgress('Analyzed image %d' % imageNum)
                t0 = t
                if self.cancelling or self.exiting:
                    self.showProgress('Cancelled')
                    break
        data = np.abs(np.stack(dataList, axis=0))
        if len(data.shape) == 5:
            data2 = np.mean(data, axis=(2))    # Averaging by other convolution channels dimension out of 2
        data = np.mean(data, axis=(1))
        # data = np.log10(data)
        self.showProgress('Gradients: %s, max %.4f (%s)' % \
                (str(data.shape), data.max(), str([int(v[0]) for v in np.where(data == data.max())])))
        self.showGradients(data, data2, True)

    # MultipleObjects means that many images or epochs will be depicted, so a logarithm should be shown
    def showGradients(self, data, data2, multipleObjects):
        stdData = None
        drawMode = 'map'
        shape = data.shape
        if len(shape) >= 3:
            # data = np.reshape(data, (shape[0] * shape[1], shape[2], shape[3]))
            restAxis = tuple(range(2, len(data.shape)))
            stdData = np.std(data, axis=restAxis)
            data = np.mean(data, axis=restAxis)
        if not data2 is None and len(data2.shape) >= 3:
            data2 = np.mean(data2, axis=tuple(range(2, len(data2.shape))))

        shape = data.shape
        if len(shape) == 2:
            drawMode = '2d'
            # if shape[1] < 50:
            #     drawMode = 'plot'
            #     data = data.flatten()
            # else:
            #     data = np.reshape(data, [data.shape[1], 1, 1])
            #     margin = 0
        else:
            data = self.getChannelsToAnalyze(data)
            margin = self.c_channelMargin

        if multipleObjects and len(data.shape) >= 2:
            data = data.transpose([1, 0] + list(range(2, len(data.shape))))
            if not stdData is None:
                stdData = stdData.transpose([1, 0] + list(range(2, len(stdData.shape))))
            if not data2 is None:
                data2 = data2.transpose([1, 0] + list(range(2, len(data2.shape))))

        print("Data transformed")
        self.clearFigure()
        ax = self.getMainSubplot()
        ax.clear()
        if drawMode == 'map':
            colCount = math.ceil(math.sqrt(data.shape[0]) * 1.15 / 2) * 2
            data = layoutLayersToOneImage(data,  #np.sqrt(activations),
                                          colCount, margin, 0)
            im = ax.imshow(data, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)
            ax.title.set_text('Sqrt')
        elif drawMode == '2d':
            if multipleObjects:
                data = np.abs(data)
                data[data < 1e-15] = 1e-15
                data = np.log10(data)
                ax.title.set_text('Log10')
            im = ax.imshow(data, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)
        else:
            ax.plot(data)

        if not stdData is None:
            ax = self.figure.add_subplot(self.gridSpec[0, 0])
            ax.clear()
            im = ax.imshow(stdData, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)
            ax.title.set_text('Std. dev.')

        ax = self.figure.add_subplot(self.gridSpec[1, 0])
        ax.clear()
        if data2 is None:
            ax.hist(data.flatten(), bins=100)
            if ax.get_ylim()[0] > ax.get_ylim()[1]:
                ax.invert_yaxis()
            ax.set_aspect('auto')
        else:
            if multipleObjects:
                data2 = np.abs(data2)
                data2[data2 < 1e-15] = 1e-15
                data2 = np.log10(data2)
            im = ax.imshow(data2, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)

        print("Plots Updated")
        self.canvas.draw()
        print("Done")

    def onShowWorstImagesPressed(self):
        try:
            self.startAction(self.onShowWorstImagesPressed)
            epochNum = self.getSelectedEpochNum()    # TODO
            firstImageCount = max(1000, self.getSelectedImageNum())
            self.netWrapper.loadState(epochNum)     # TODO: to wrap into a NetWrapper's method
            (losses, predictions) = self.netWrapper._getNet().getImageLosses(1, firstImageCount)

            arr = np.arange(1, firstImageCount + 1)
            losses = np.stack([arr, losses], axis=0)
            sortedVals = losses[:, losses[1, :].argsort()]

            imageNums = np.flip(sortedVals[0, -300:])
            colCount = 20
            imageData = []
            for imageNum in imageNums:
                curImageData = self.imageDataset.getImage(int(imageNum), 'cropped')

                if curImageData.max() > 1.01:
                    if imageNum <= 255:
                        curImageData[-1][-1] = imageNum
                else:
                    if imageNum <= 999:
                        curImageData[-1][-1] = imageNum / 1000.0

                imageData.append(curImageData)
            imageData = np.stack(imageData, axis=0)
            imageData = layoutLayersToOneImage(imageData, colCount, 1, imageData.min())

            self.clearFigure()

            ax = self.getMainSubplot()
            self.showImage(ax, imageData)
            # im = ax.imshow(imageData, cmap='plasma')

            lossData = np.expand_dims(np.expand_dims(np.flip(sortedVals[1, -300:]), 1), 2)
            lossData = layoutLayersToOneImage(lossData, colCount, 0, lossData.min())

            ax = self.figure.add_subplot(self.gridSpec[1, 0])
            ax.clear()
            im = ax.imshow(lossData, cmap='plasma')
            colorBar = self.figure.colorbar(im, ax=ax)
            self.canvas.draw()

            self.showProgress("Images: %s" % ', '.join([str(int(num)) for num in imageNums[:10]]))
            np.savetxt('Data/WorstImageNums.txt', imageNums, fmt='%d', delimiter='')
            # with open('Data/WorstImageNums.dat', 'wb') as file:
            #     pickle.dump(imageNums, file)
        except Exception as ex:
            self.showProgress("Error: %s" % str(ex))


    def onCorrelationAnalyzisPressed(self):
        self.startAction(self.onCorrelationAnalyzisPressed)
        self.doCorrelationAnalyzis(False)

    def onCorrelationToOtherModelPressed(self):
        self.startAction(self.onCorrelationToOtherModelPressed)
        self.doCorrelationAnalyzis(True)

    def doCorrelationAnalyzis(self, toOtherModel):
        import CorrelatAnalyzis

        calculator = CorrelatAnalyzis.CCorrelationsCalculator(self, self.activationCache, self.netWrapper)
        options = calculator
        epochNum = self.getSelectedEpochNum()
        firstImageCount = max(1000 if self.netWrapper.name == 'mnist' else 200,
                              self.getSelectedImageNum())
        layerName = self.blockComboBox.currentText()
        options.towerCount = self.netWrapper.getTowerCount()

        imagesActs = self.netWrapper.getImagesActivations_Batch(
                layerName, range(1, firstImageCount + 1), epochNum)
        self.showProgress('Activations: %s, max %.4f (%s)' % \
                (str(imagesActs.shape), imagesActs.max(),
                 str([int(v[0]) for v in np.where(imagesActs == imagesActs.max())])))
        # ests = self.getEstimations(imagesActs)

        self.clearFigure()
        # ax = self.getMainSubplot()

        if 1:
            ax = self.figure.add_subplot(self.gridSpec[0, 0])
            varianceDistrs = calculator.getTowersVarianceDistributions(imagesActs)
            for i in range(varianceDistrs.shape[0]):
                ax.plot(varianceDistrs[i], label='%d' % i)
            ax.legend()
            ax.title.set_text('Variance distribution')

        try:
            if toOtherModel:
                imagesActs2 = self.loadOtherModelActivations(firstImageCount, layerName)
                calculator.show2ModelsCorrelations(imagesActs, imagesActs2)
            else:
                calculator.showTowersCorrelations(imagesActs)
        except Exception as ex:
            self.showProgress("Error on correlations: %s" % str(ex))

        self.canvas.draw()
        self.saveCorrelationsImage(epochNum, firstImageCount, layerName)

    def saveCorrelationsImage(self, epochNum, firstImageCount, layerName):
        dirName = 'Data/%s_Correlations_%dImages' % \
                 (layerName, firstImageCount)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        fileName = '%s/Corr_%s_Epoch%d.png' % \
                 (dirName, layerName, epochNum)

        prevDpi = matplotlib.rcParams['savefig.dpi']
        self.figure.savefig(fileName, format='png', dpi=150)
        # try:
            # matplotlib.rcParams['savefig.dpi'] = 600

    def saveActsForCorrelations(self):
        epochNum = self.getSelectedEpochNum()
        firstImageCount = max(2000 if self.netWrapper.name == 'mnist' else 500,
                              self.getSelectedImageNum())
        if not os.path.exists('Data/Model2'):
            os.makedirs('Data/Model2')
        for layerName in ['conv_2', 'conv_3', 'conv_4', 'dense_1']:
            imagesActs = self.netWrapper.getImagesActivations_Batch(
                layerName, range(1, firstImageCount + 1), epochNum)
            with open('Data/Model2/Activations_%s.dat' % layerName, 'wb') as file:
                pickle.dump(imagesActs, file)
            self.showProgress('%s activations saved: %s' % \
                (layerName, str(imagesActs.shape)))

    def loadOtherModelActivations(self, firstImageCount, layerName):
        with open('Data/Model2/Activations_%s.dat' % layerName, 'rb') as file:
            imagesActs = pickle.load(file)
        return imagesActs[:firstImageCount]


    def onChanSpinBoxValueChanged(self):
        if self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed]:
            self.lastAction()

    # Main method currently that updates data in accordance with entered image number/channel number
    # and previously pressed displaying buttons
    def onSpinBoxValueChanged(self):
        try:
            if self.lastAction in [self.onShowImagePressed, self.onShowActivationsPressed, self.onShowActTopsPressed]:
                t0 = datetime.datetime.now()
                self.onShowActivationsPressed()
                self.onShowImagePressed()
                # self.showProgress('2 operations: %.1f ms' % ((datetime.datetime.now() - t0).total_seconds() * 1000))

                # self.lastAction()
            elif self.lastAction in [self.onShowChanActivationsPressed, self.onShowSortedChanActivationsPressed,
                                     self.onShowGradientsPressed, self.onGradientsByImagesPressed]:
                self.lastAction()
        except Exception as ex:
            self.showProgress("Error: %s" % str(ex))

    def onEpochComboBoxChanged(self):
        # if getCpuCoreCount() > 4 and
        if self.lastAction in [self.onShowMultActTopsPressed, self.onGradientsByImagesPressed,
                               self.onShowActEstByImagesPressed, self.onShowWorstImagesPressed,
                               self.onCorrelationAnalyzisPressed, self.onCorrelationToOtherModelPressed]:
            self.lastAction()
        else:
            self.onSpinBoxValueChanged()   # onDisplayPressed()

    def onBlockChanged(self):
        if self.lastAction in [self.onShowImagePressed,
                               self.onShowActivationsPressed, self.onShowActTopsPressed]:
            self.onSpinBoxValueChanged()
        elif self.lastAction in [self.onShowMultActTopsPressed, self.onShowActEstByImagesPressed,
                                 self.onShowGradientsPressed, self.onGradientsByImagesPressed,
                                 self.onCorrelationAnalyzisPressed, self.onCorrelationToOtherModelPressed,
                                 self.showWeightsImage2] and \
                self.getSelectedEpochNum() and self.getSelectedEpochNum() > 0:
            self.lastAction()


    class TLearnOptions:
        def __init__(self, learnRate):
            self.learnRate = learnRate
            self.trainImageNums = None
            self.additTrainImageCount = 0

    def onDoItersPressed(self, iterCount, trainImageNums=None):
        # epochIterCount = DeepOptions.trainSetSize / DeepOptions.batchSize
        # c_callIterCount = int(epochIterCount)  # 400

        self.cancelling = False
        curEpochNum = self.getSelectedEpochNum()
        if curEpochNum < 0:
            savedEpochNums = self.netWrapper.getSavedNetEpochs()
            if savedEpochNums:
                curEpochNum = savedEpochNums[-1]
        if curEpochNum >= 0 and self.netWrapper.curEpochNum != curEpochNum:
            self.netWrapper.loadState(curEpochNum)

        # if self.weightsReinitEpochNum is None:
        #     restoreRestEpochCount = 0
        # else:
        #     restoreRestEpochCount = 40 - (curEpochNum - self.weightsReinitEpochNum)
        callback = QtMainWindow.TLearningCallback(self, curEpochNum)
        # callback.learnRate = float(self.learnRateEdit.text())
        options = QtMainWindow.TLearnOptions(float(self.learnRateEdit.text()))
        if not trainImageNums is None:
            options.trainImageNums = np.array(trainImageNums, dtype=int)
            options.additTrainImageCount = max(500, self.imageNumEdit.value()) - len(trainImageNums)

        infoStr = self.netWrapper.doLearning(iterCount, options, callback)

        self.loadNetStateList()
        self.epochComboBox.setCurrentIndex(self.epochComboBox.count() - 1)
        self.iterNumLabel.setText('Epoch %d finished' % self.netWrapper.curEpochNum)

        # restIterCount = iterCount
        # while restIterCount > 0:
        #     learnResult = self.netTrader.doLearning(
        #             c_callIterCount if restIterCount > c_callIterCount else restIterCount)
        #     restIterCount -= c_callIterCount
        #     trainIterNum = self.netTrader.getTrainIterNum()
        #     self.iterNumLabel.setText('Iteration %d' % trainIterNum)
        #     if self.exiting:
        #         return
        #     app.processEvents()
        #
        #     # epochNum = trainIterNum / epochIterCount
        #     # print("Epoch * 3 %%: ", (epochNum * 3) % (DeepOptions.learningRate.circularStepEpochCount * 3))
        #     # if int(epochNum) % (DeepOptions.learningRate.circularStepEpochCount * 3) == 0:
        #     #     print("Saving state after %d epochs (%d iterations)" % (epochNum, trainIterNum))
        #     #     DeepMain.MainWrapper.saveState(self, "States/State_%d_%05d_%f/state.dat" % \
        #     #             (epochNum, trainIterNum, learnResult))

        if self.lastAction in [self.onShowMultActTopsPressed]:
            self.lastAction()
        else:
            self.onSpinBoxValueChanged()   # onDisplayPressed()
        self.showProgress(infoStr, False)

    # def onDoItersOnWorstPressed(self, iterCount):
    def onSpecialDoItersPressed(self, iterCount):
        if 0:
            imageNums = np.loadtxt('Data/SelectedWorstImageNums.txt', dtype=int, delimiter='\n')
            imageNums = list(set(imageNums))
            print('%d unique worst images numbers read' % len(imageNums))
            # additImageCount = max(1000, self.imageNumEdit.value()) - len(imageNums)
            # if additImageCount > 0:
            #     imageNums = np.concatenate([imageNums, np.random.randint(low=1, high=20000, size=additImageCount)])
        else:
            imageNums = np.loadtxt('Data/ImageNums/DistantImageNums1_conv_5.txt', dtype=int, delimiter='\n')
            imageNums = list(set(imageNums))
            print('%d unique images numbers read' % len(imageNums))

        self.onDoItersPressed(iterCount, imageNums)

    # Expects number like 9990 at channel number spin box (each 0 to 9 - weight from 0 to 100%)
    def onSetTowerWeightsPressed(self):
        try:
            # curWeights = self.netWrapper.getMultWeights('tower_weights')
            curWeights = self.netWrapper.getVariableValue('tower_weights')
            intValue = self.chanNumEdit.value()
            strValue = '%04d' % intValue
            newWeights = [int(ch) / 9.0 for ch in strValue]
            newWeights = ([0] * (len(curWeights) - len(newWeights))) + newWeights
            print('Tower weights: %s, replacing with %s' % (str(curWeights), str(newWeights)))
            self.netWrapper.setVariableValue('tower_weights', newWeights)
            self.activationCache.clear()
        except Exception as ex:
            self.infoLabel.setText('Error on weights set: %s' % str(ex))

    class TLearningCallback(MnistNetVisWrapper.CBaseLearningCallback):
        def __init__(self, parent, curEpochNum):
            super(QtMainWindow.TLearningCallback, self).__init__()
            self.parent = parent
            self.curEpochNum = curEpochNum
            # self.restoreRestEpochCount = restoreRestEpochCount
            self.passedEpochCount = 0
            self.trainIterNum = -1

        # def __call__(self, infoStr):

        def onBatchEnd(self, trainIterNum, logs):
            self.logs = logs
            self.trainIterNum = trainIterNum

            reinitEpochNum = self.parent.weightsReinitEpochNum
            if reinitEpochNum is not None and reinitEpochNum <= self.curEpochNum and \
                    self.curEpochNum - reinitEpochNum < 10:
                self.parent.restoreReinitedNeirons(True)
            super(QtMainWindow.TLearningCallback, self).onBatchEnd(logs, trainIterNum)

        def onSecondPassed(self):
            self.logs.pop('batch', None)
            self.logs.pop('size', None)
            self.parent.showProgress('Iter %d: %s' %
                    (self.trainIterNum, str(self.logs)))
            if self.parent.cancelling or self.parent.exiting:
                self.parent.netWrapper.cancelling = True

            reinitEpochNum = self.parent.weightsReinitEpochNum
            if reinitEpochNum is not None and reinitEpochNum <= self.curEpochNum and \
                    self.curEpochNum - reinitEpochNum >= 10 and self.curEpochNum - reinitEpochNum < 40:
                self.parent.restoreReinitedNeirons()
            # if self.restoreRestEpochCount > 0:
            #     self.parent.restoreReinitedNeirons()

        def onEpochEnd(self, curEpochNum, infoStr):
            self.parent.showProgress(infoStr)
            # self.restoreRestEpochCount -= 1
            self.curEpochNum = curEpochNum
            self.passedEpochCount += 1

            self.lastUpdateTime = datetime.datetime.now() + datetime.timedelta(seconds=2)
                # A hacking of super-class in order to make epoch info visible longer


    def onTestPress(self):
        # self.loadState()
        pass

    def onLoadStatePressed(self):
        self.loadState(self.getSelectedEpochNum())

    def onReinitWorstNeironsPressed(self):
        self.startAction(self.onReinitWorstNeironsPressed)
        curEpochNum = self.getSelectedEpochNum()
        if curEpochNum < 0:
            curEpochNum = self.netWrapper.getSavedNetEpochs()[-1]
        self.weightsBeforeReinit = dict()
        self.weightsAfterReinit = dict()
        self.weightsReinitInds = dict()

        if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper):
            self.reinitWorstNeirons_Random(curEpochNum)
        else:  # if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper3_Towers):
            # self.reinitWorstNeirons_OneLayerTowers(curEpochNum)
            self.reinitWorstNeirons_SeparatedTowers(curEpochNum)

        self.weightsReinitEpochNum = curEpochNum
        self.infoLabel.setText('Worst neirons reinitialized')

    def reinitWorstNeirons_Random(self, curEpochNum):
        for layerName in self.netWrapper.getComponentNetLayers():
            gradients = self.netWrapper.getGradients(layerName, 1, 500, curEpochNum)
            weights = self.netWrapper.getMultWeights(layerName)

            std0 = np.std(weights)
            gradients2 = np.square(gradients)
            resetShape = list(gradients.shape)
            self.weightsBeforeReinit[layerName] = weights
            if layerName[:4] == 'conv':
                meanGradients = np.mean(gradients2, axis=(0, 2, 3))
                sortedInds = meanGradients.argsort()
                indsToChange = sortedInds[4 : ]
                others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)], :, :]
                othersStdDev = np.std(others)
                resetShape[1] = len(indsToChange)
                weights[:, indsToChange, :, :] = \
                    np.random.normal(scale=othersStdDev / 16, size=resetShape)
            elif layerName[:5] == 'dense':
                meanGradients = np.mean(gradients2, axis=(0, ))
                sortedInds = meanGradients.argsort()
                if 0:
                    indsToChange = sortedInds[12 if layerName == 'dense_1' else 1 : ]
                    others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)]]
                else:
                    indsToChange = sortedInds
                    others = weights
                othersStdDev = np.std(others)
                resetShape[1] = len(indsToChange)
                weights[:, indsToChange] = \
                    np.random.normal(scale=othersStdDev / 16, size=resetShape)
            else:
                indsToChange = None
                del self.weightsBeforeReinit[layerName]

            if not indsToChange is None:
                self.netWrapper.setMultWeights(layerName, weights)
                self.weightsReinitInds[layerName] = indsToChange
                self.weightsAfterReinit[layerName] = weights
                print("%s reinit: std. %.9f - %.9f, neirons %s" % \
                      (layerName, std0, np.std(weights),
                       ', '.join(sorted([str(i) for i in indsToChange]))))

    def reinitWorstNeirons_OneLayerTowers(self, curEpochNum):
        towerCount = self.netWrapper.getMultWeights('conv_1').shape[1]          # E.g., 80, weights - 1 * 80 * 5 * 5
        gradients = self.netWrapper.getGradients('conv_3', 1, 500, curEpochNum)    # E.g. (80 * 4) * 2 * 3 * 3
        gradients2 = np.reshape(np.square(gradients), (towerCount, -1, gradients.shape[2], gradients.shape[3]))
        meanGradients = np.mean(gradients2, axis=(1, 2, 3))
        sortedInds = meanGradients.argsort()
        towerIndsToChange = sortedInds[8 : ]
        for layerName in self.netWrapper.getNetLayersToVisualize():
            gradients = self.netWrapper.getGradients(layerName, 1, 500, curEpochNum)
            weights = self.netWrapper.getMultWeights(layerName)

            std0 = np.std(weights)
            resetShape = list(gradients.shape)
            self.weightsBeforeReinit[layerName] = weights
            if layerName == 'conv_1':
                indsToChange = towerIndsToChange
                others = weights[:, sortedInds[: gradients.shape[1] - len(indsToChange)], :, :]
                othersStdDev = np.std(others)
                resetShape[1] = len(indsToChange)
                weights[:, indsToChange, :, :] = \
                    np.random.normal(scale=othersStdDev, size=resetShape)
            elif layerName[:4] == 'conv':
                towerWidth = weights.shape[0] // towerCount
                indsToChange = []
                for i in range(towerWidth):
                    indsToChange.append(towerIndsToChange * towerWidth + i)
                indsToChange = np.concatenate(indsToChange, axis=0)

                others = weights[sortedInds[ : gradients.shape[0] - len(indsToChange)], :, :, :]
                othersStdDev = np.std(others)
                resetShape[0] = len(indsToChange)
                weights[indsToChange, :, :, :] = \
                    np.random.normal(scale=othersStdDev, size=resetShape)
            elif layerName[:5] == 'dense':
                gradients2 = np.square(gradients)
                meanGradients = np.mean(gradients2, axis=(0, ))
                sortedInds = meanGradients.argsort()
                if 0:
                    indsToChange = sortedInds[12 if layerName == 'dense_1' else 1 : ]
                    others = weights[:, sortedInds[ : gradients.shape[1] - len(indsToChange)]]
                else:
                    indsToChange = sortedInds
                    others = weights
                othersStdDev = np.std(others)
                resetShape[1] = len(indsToChange)
                weights[:, indsToChange] = \
                    np.random.normal(scale=othersStdDev, size=resetShape)
            else:
                indsToChange = None
                del self.weightsBeforeReinit[layerName]

            if not indsToChange is None:
                self.netWrapper.setMultWeights(layerName, weights)
                self.weightsReinitInds[layerName] = indsToChange
                self.weightsAfterReinit[layerName] = weights
                print("%s reinit: std. %.9f - %.9f, neirons %s" % \
                      (layerName, std0, np.std(weights),
                       ', '.join(sorted([str(i) for i in indsToChange]))))

    # For CMnistModel3_Towers
    def reinitWorstNeirons_SeparatedTowers(self, curEpochNum):
        towerInd = 0
        towerMeans = []
        while True:
            try:
                gradients = self.netWrapper.getGradients('conv_2_%d' % towerInd, 1, 500, curEpochNum)
            except Exception as ex:
                break   # No more towers

            gradients2 = np.square(gradients)         # E.g. 40 * 24 * 3 * 3
            meanGradients = np.mean(gradients2) # , axis=(1, 2, 3))
            towerMeans.append(meanGradients)
            towerInd += 1
        if towerInd <= 0:
            raise Exception('No towers found')

        towerMeans = np.stack(towerMeans)
        sortedTowerInds = towerMeans.argsort()
        towerIndsToChange = sortedTowerInds[4:]
        towersToPreserveInds = sortedTowerInds[:4]
        for towerInd in towerIndsToChange:
            for layerName in ['conv_1_%d' % towerInd, 'conv_2_%d' % towerInd]:
                weights = self.netWrapper.getMultWeights(layerName)
                std0 = np.std(weights)
                resetShape = list(weights.shape)
                # self.weightsBeforeReinit[layerName] = weights

                others = weights # [:, sortedInds[: gradients.shape[1] - len(indsToChange)], :, :]
                othersStdDev = np.std(others)
                weights[:, :, :, :] = \
                    np.random.normal(scale=othersStdDev / 16, size=resetShape)
                self.netWrapper.setMultWeights(layerName, weights)

        if 1:
            layerName = 'conv_3'
            # gradients = self.netWrapper.getGradients(layerName, 500, curEpochNum)
            weights = self.netWrapper.getMultWeights(layerName)

            std0 = np.std(weights)
            # gradients2 = np.square(gradients)
            resetShape = list(weights.shape)

            indsToChange = []
            assert weights.shape[0] % len(sortedTowerInds) == 0
            towerWidth = weights.shape[0] // len(sortedTowerInds)
            for towerInd in towerIndsToChange:
                indsToChange += range(towerInd * towerWidth, (towerInd + 1) * towerWidth)
            # meanGradients = np.mean(gradients2, axis=(0, 2, 3))
            # sortedInds = meanGradients.argsort()
            resetShape[0] = len(indsToChange)
            indsToPreserve = set(range(len(indsToChange)))
            indsToPreserve = list(indsToPreserve.difference(indsToChange))

            # self.weightsBeforeReinit[layerName] = weights
            weights[indsToChange, :, :, :] = \
                np.random.normal(scale=othersStdDev / 16, size=resetShape)
            weights[indsToPreserve, :, :, :] *= 4
            # self.weightsReinitInds[layerName] = indsToChange
            # self.weightsAfterReinit[layerName] = weights

        keepLayerNames = ['dense_1', 'dense_2']
        for towerInd in towersToPreserveInds:
            for layerName in ['conv_1_%d' % towerInd, 'conv_2_%d' % towerInd]:
                keepLayerNames.append(layerName)
        for layerName in keepLayerNames:
            weights = self.netWrapper.getMultWeights(layerName)
            self.weightsBeforeReinit[layerName] = weights
            self.weightsReinitInds[layerName] = []
            self.weightsAfterReinit[layerName] = weights
            # print("%s reinit: std. %.9f - %.9f, neirons %s" % \
            #       (layerName, std0, np.std(weights),
            #        ', '.join(sorted([str(i) for i in indsToChange]))))

    def restoreReinitedNeirons(self, veryStrong=False):
        # print('restoreReinitedNeirons %s' % ('very strong' if veryStrong else ''))
        if isinstance(self.netWrapper, MnistNetVisWrapper.CMnistVisWrapper):
            indicesAxis = 1
        else:
            indicesAxis = 0

        for layerName in self.netWrapper.getComponentNetLayers():
            if layerName in self.weightsBeforeReinit:
                # if veryStrong and layerName[:4] == 'conv':
                #     self.netWrapper.setMultWeights(layerName, self.weightsAfterReinit[layerName])
                #     continue

                weights = self.netWrapper.getMultWeights(layerName)
                indsToChange = self.weightsReinitInds[layerName]
                indsToPreserve = set(range(len(indsToChange)))
                indsToPreserve = list(indsToPreserve.difference(indsToChange))
                print("Restoring %d weights at %s" % (len(indsToPreserve), layerName))

                if layerName[:4] == 'conv':
                    # if layerName == 'conv_3':
                    #     w
                    if indicesAxis == 1 or layerName == 'conv_1':    # Restoring after reinitWorstNeirons_Random
                                                                     # or conv_1 after reinitWorstNeirons_OneLayerTowers
                        weights[:, indsToPreserve, :, :] = \
                            self.weightsBeforeReinit[layerName][:, indsToPreserve, :, :]
                    else:                                            # Restoring after reinitWorstNeirons_OneLayerTowers
                        weights[indsToPreserve, :, :, :] = \
                            self.weightsBeforeReinit[layerName][indsToPreserve, :, :, :]
                    self.netWrapper.setMultWeights(layerName, weights)
                elif layerName[:5] == 'dense':
                    weights[:, indsToPreserve] = \
                        self.weightsBeforeReinit[layerName][:, indsToPreserve]
                    self.netWrapper.setMultWeights(layerName, weights)


    def saveState(self):
        try:
            self.netWrapper.saveState()
        except Exception as ex:
            self.showProgress("Error in saveState: %s" % str(ex))

    def loadNetStateList(self):
        try:
            savedNetEpochs = self.netWrapper.getSavedNetEpochs()
            self.epochComboBox.clear()
            self.epochComboBox.addItem('---  All  ---')
            for epochNum in savedNetEpochs:
                self.epochComboBox.addItem('Epoch %d' % epochNum)
        except Exception as ex:
            self.showProgress("Error in loadState: %s" % str(ex))

    def loadState(self, epochNum=None):
        try:
            if epochNum is None:
                epochNum = -1      # When there is only one file and its epoch is unknown
            self.netWrapper.loadState(epochNum)
            # self.netWrapper.loadCacheState()
            self.iterNumLabel.setText('Epoch %d' % self.netWrapper.curEpochNum)
        except Exception as ex:
            self.showProgress("Error in loadState: %s" % str(ex))

    def loadCacheState(self):
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
            # print("Exception in mouseMove: ", str(ex))
            pass  # Sometimes we are getting MouseMoveEvent object without x and y

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

        # self.clearFigure()
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
        self.clearFigure()
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

    def onIncreaseLearningRatePressed(self):
        self.net.changeRate(2)
        self.infoLabel.setText('Learning rate increased')

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
mainWindow = None

def sigint_handler(*args):
    # global app
    global mainWindow

    """Handler for the SIGINT signal."""
    sys.stderr.write('Exiting\n')
    # app.exiting = True
    mainWindow.exiting = True
    QtGui.QApplication.quit()

if __name__ == "__main__":

    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    # List ALL tensors.
    # print_tensors_in_checkpoint_file(file_name='State/checkpoint', tensor_name='', all_tensors=True)


    # app.use('pyside')
    app = QtGui.QApplication(sys.argv)
    # app.exiting = False

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
        # mainWindow.loadCacheState()

        try:
            # mainWindow.loadState(mainWindow.getSelectedEpochNum())
            mainWindow.blockComboBox.setCurrentIndex(0)
            # mainWindow.netWrapper.doubleLayerWeights(['conv_21'])
            mainWindow.showWeightsImage2()
            mainWindow.blockComboBox.setCurrentIndex(1)
            mainWindow.showWeightsImage2()
            mainWindow.saveState()
            # mainWindow.netWrapper.getGradients()
            # MnistNetVisWrapper.getGradients(mainWindow.netWrapper.net.model)
            # mainWindow.onDoItersPressed(1)
            # mainWindow.onReinitWorstNeironsPressed()

            # mainWindow.onShowImagePressed()
            # mainWindow.onDisplayIntermResultsPressed()
            # mainWindow.onDisplayPressed()
            # mainWindow.onShowActivationsPressed()
            # mainWindow.onShowMultActTopsPressed()
            # mainWindow.onShowSortedChanActivationsPressed()
            # mainWindow.onSpinBoxValueChanged()
            # mainWindow.calcMultActTops_MultiThreaded()
            # mainWindow.onGradientsByImagesPressed()
            # mainWindow.onShowWorstImagesPressed()
            # mainWindow.onShowImagesWithTSnePressed()
            # mainWindow.onCorrelationToOtherModelPressed()
            # mainWindow.netWrapper.setLearnRate(mainWindow.netWrapper.getRecommendedLearnRate())

            pass
        except Exception as ex:
            print('Exception in main procedure: %s' % str(ex))
            raise

        # for i in range(10):
        #     try:
        #         mainWindow.showWeightsImage2()
        #     except Exception as ex:
        #         print('Exception in main procedure: %s' % str(ex))
        #         raise
    else:
        mainWindow.fastInit()
    # mainWindow.paintRect()
    sys.exit(app.exec_())
    # mainWindow.saveState()
