# import concurrent.futures
# import copy
import datetime
import matplotlib
matplotlib.use('AGG')
matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvas
    # +FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
# from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
# from matplotlib import cm
# import pickle
import math
import numpy as np
# import PyQt4.Qt
# from PyQt4 import QtCore, QtGui
import os
# import random
import sys
# import time
# import queue
# import _thread

prevSysPath = sys.path
sys.path.append("External/svcca")
sys.path.append("../External/svcca")
import cca_core

from MyUtils import *
# from VisUtils import *

# class TMultActOpsOptions:
#     topCount = 25
#     oneImageMaxTopCount = 4
#     minDist = 3
#     batchSize = 16 * getCpuCoreCount()
#     embedImageNums = False
#     c_channelMargin = 2
#     c_channelMargin_Top = 5

class CCorrelationsCalculator:
    # AllEpochs = -2

    def __init__(self, mainWindow, activationCache, netWrapper=None):
        self.mainWindow = mainWindow
        self.netWrapper = netWrapper
        self.activationCache = activationCache
        self.imageDataset = self.netWrapper.getImageDataset()

        # Options:
        self.towerCount = 1

    def flattenConvActivations(self, imagesActs):
        acts = imagesActs.transpose([1, 0] + list(range(2, len(imagesActs.shape))))
        shape2 = 1
        for count in acts.shape[1:]:
            shape2 *= count
        return np.reshape(acts, [acts.shape[0], shape2])

    def showTowersCorrelations(self, imagesActs):
        towerChannelCount = imagesActs.shape[1] // self.towerCount
        for towerInd1 in range(1): # self.towerCount - 1):
            tower1Acts = self.flattenConvActivations(
                    imagesActs[:, towerInd1 * towerChannelCount : (towerInd1 + 1) * towerChannelCount])
            _, s, v = np.linalg.svd(tower1Acts - np.mean(tower1Acts, axis=1, keepdims=True), full_matrices=False)
            for towerInd2 in range(towerInd1 + 1, self.towerCount):
                tower2Acts = self.flattenConvActivations(
                        imagesActs[:, towerInd2 * towerChannelCount : (towerInd2 + 1) * towerChannelCount])
                # for i in range(25, tower1Acts.shape[1], 50):
                #     results = cca_core.get_cca_similarity(tower1Acts[:, :i], tower2Acts[:, :i], epsilon=1e-10)
                #     print('%d ok' % i)
                result = cca_core.get_cca_similarity(tower1Acts, tower2Acts, epsilon=1e-10)
                if towerInd2 == 1:
                    print(result.keys)
                # ax = self.mainWindow.figure.add_subplot(5, 4, towerInd1 * 4 + towerInd2 + 2)
                ax = self.mainWindow.getMainSubplot()
                ax.plot(result['cca_coef1'])

    def show2ModelsCorrelations(self, imagesActs, imagesActs2):
        towerChannelCount1 = imagesActs.shape[1] // self.towerCount
        towerChannelCount2 = imagesActs2.shape[1] // self.towerCount
        areDimensionsDifferent = (imagesActs.shape[2:] != imagesActs2.shape[2:])
        for towerInd1 in range(2): # self.towerCount - 1):
            data = imagesActs[:, towerInd1 * towerChannelCount1 : (towerInd1 + 1) * towerChannelCount1]
            if areDimensionsDifferent:
                means = np.mean(data, axis=tuple(range(2, len(imagesActs.shape))))
                stds = np.std(data, axis=tuple(range(2, len(imagesActs.shape))))
                data = np.stack([means, stds], axis=2)
            tower1Acts = self.flattenConvActivations(data)

            for towerInd2 in range(self.towerCount):
                data = imagesActs2[:, towerInd2 * towerChannelCount2 : (towerInd2 + 1) * towerChannelCount2]
                if areDimensionsDifferent:
                    means = np.mean(data, axis=tuple(range(2, len(imagesActs.shape))))
                    stds = np.std(data, axis=tuple(range(2, len(imagesActs.shape))))
                    data = np.stack([means, stds], axis=2)
                tower2Acts = self.flattenConvActivations(data)

                # for i in range(25, tower1Acts.shape[1], 50):
                #     results = cca_core.get_cca_similarity(tower1Acts[:, :i], tower2Acts[:, :i], epsilon=1e-10)
                #     print('%d ok' % i)
                result = cca_core.get_cca_similarity(tower1Acts, tower2Acts, epsilon=1e-10)
                if towerInd2 == 1:
                    print(result.keys)
                # ax = self.mainWindow.figure.add_subplot(5, 4, towerInd1 * 4 + towerInd2 + 2)
                ax = self.mainWindow.getMainSubplot()
                ax.plot(result['cca_coef1'])

    def getTowersVarianceDistributions(self, imagesActs):
        towerChannelCount = imagesActs.shape[1] // self.towerCount
        data = []
        for towerInd1 in range(self.towerCount):
            tower1Acts = self.flattenConvActivations(
                    imagesActs[:, towerInd1 * towerChannelCount : (towerInd1 + 1) * towerChannelCount])
            _, s, v = np.linalg.svd(tower1Acts - np.mean(tower1Acts, axis=1, keepdims=True), full_matrices=False)
            data.append(s)
        return np.stack(data, axis=0)

