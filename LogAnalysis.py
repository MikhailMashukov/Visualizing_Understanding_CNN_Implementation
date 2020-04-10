# Should be run with C:\Program Files\Python35

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import os
import pandas as pd
try:
    from matplotlib.backends.backend_qt4agg import FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
    from PyQt4.QtGui import (QApplication, QColumnView, QFileSystemModel,
                     QSplitter, QTreeView, QItemSelection)
    from PyQt4.QtCore import QDir, Qt
    from PyQt4 import QtCore, QtGui
except Exception as ex:
    print("Warning: no Qt")
import re
import signal
import sys

app = None
mainWindow = None

# for m in sys.modules:
#     print(m)

if 'PyQt4.QtGui' in sys.modules:
  class LogAnalysisQtMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        # app = QApplication(sys.argv)
        self.displayedFilePath = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Neur. Net. Logs Analysis')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)       # Additional _main object - workaround for not appearing FigureCanvas

        # splitter = QSplitter(self._main)

        # self.infoLabel = QtGui.QLabel(self)
        # self.infoLabel.setMinimumWidth(350)
        self.statusBar = QtGui.QStatusBar()
        self.setStatusBar(self.statusBar)

        splitter = QtGui.QHBoxLayout(self._main)
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setRootPath(os.path.split(sys.argv[0])[0])
        # self.fileSystemModel.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.fileSelectView = QTreeView(self)
        self.fileSelectView.setModel(self.fileSystemModel)
        self.fileSelectView.setMaximumWidth(350)
        self.fileSelectView.setColumnWidth(0, 400)
        self.fileSelectView.setColumnWidth(1, 70)
        self.fileSelectView.hideColumn(2)
        # self.fileSelectView.currentChanged.connect(self.onFileSelected)
        self.fileSelectView.clicked.connect(self.onFileSelected)
        self.fileSelectView.selectionModel().selectionChanged.connect(self.onFileSelected)
        # self.fileSelectView.setCurrentIndex(self.fileSystemModel.index(os.path.split(sys.argv[0])[0] + '/log'))
        splitter.addWidget(self.fileSelectView)

        # for i in range(1,5):
        #     button = QtGui.QPushButton('h%d' % i, self)
        #     def h(fileSelectView, i):
        #         fileSelectView.hideColumn(i)
        #     button.clicked.connect(lambda num=i: h(self.fileSelectView, num))
        #         # Num=i passes false, without it, all lambdas receive last value of i
        #     splitter.addWidget(button)
        # def createAdder(x):     # Maybe this would help
        #     return lambda y: y + x
        # adders = [createAdder(i) for i in range(4)]

        curVertWidget = QtGui.QVBoxLayout()
        self.figure = Figure(figsize=(5, 3))
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # self.canvas.mpl_connect('motion_notify_event', self.mouseMoveEvent)
        # self.figure.canvas.mpl_connect('pick_event', self.onFigurePick)
        # self.canvas.mpl_connect('button_press_event', self.mousePressEvent)
        # self.canvas.mpl_connect('button_release_event', self.mouseReleaseEvent)
        curVertWidget.addWidget(self.canvas)

        # splitter.addToolBar(QtCore.Qt.RightToolBarArea , NavigationToolbar(self.canvas, self))
        # self.addToolBar(QtCore.Qt.RightToolBarArea , NavigationToolbar(self.canvas, self))
        curVertWidget.addWidget(NavigationToolbar(self.canvas, self.canvas))
        # curHorizWidget.addWidget(self.canvas)
        splitter.addLayout(curVertWidget)
        # splitter.addWidget(self.canvas)

    def clearFigure(self):
        self.figure.clear()
        self.mainSubplotAxes = None

    def getMainSubplot(self):
        if not hasattr(self, 'mainSubplotAxes') or self.mainSubplotAxes is None:
            self.mainSubplotAxes = self.figure.add_subplot(111)  # add_subplot(self.gridSpec[:, 1])
        # self.mainSubplotAxes.plot([3,3,3],[4,5,6])
        return self.mainSubplotAxes

    def selectFile(self, filePath):
        self.fileSelectView.setCurrentIndex(self.fileSystemModel.index(filePath))

    def onFileSelected(self, index):
        if isinstance(index, QItemSelection):
            indices = index.indexes()
            if not indices:
                return
            for ind in index.indexes():
                print(self.fileSystemModel.filePath(ind))
            index = indices[0]
            # index = self.fileSelectView.selectedIndexes()
        filePath = self.fileSystemModel.filePath(index)
        print('Selected file', filePath)
        if self.displayedFilePath != filePath:
            self.showFileAnalysis(filePath)

    def showFileAnalysis(self, filePath):
        if os.path.getsize(filePath) > 1 << 20:
            self.statusBar.showMessage('File is too big')
            return
        try:
            logTable = getLogAsTable(filePath)
            # print(logTable)

            # self.figure.clear()
            ax = self.getMainSubplot()
            ax.clear()
            ext = analyzeColumn(ax, logTable, 'trainLoss')
            analyzeColumn(ax, logTable, 'valLoss')
            ax.set_ylim([ext[1], ext[3]])
            ax.legend()
            # plt.show()
            self.canvas.draw()
            self.displayedFilePath = filePath
            self.statusBar.showMessage('Info for %d epochs displayed' % logTable.shape[0])
        except Exception as ex:
            self.statusBar.showMessage('Error: %s' % str(ex))


def getLogAsTable(srcFilePath):
    table = []
    fieldNames = ['epochNum', 'trainLoss', 'trainAcc', 'valLoss', 'valAcc']
    with open(srcFilePath, 'r') as file:
        preciseCorrection = False
        epochNum = 0
        for line in file:
            # Parsing "- 9s - loss: 9.9986e-04 - accuracy: 0.0000e+00 - val_loss: 9.9930e-04 - val_accuracy: 0.0000e+00"
            match = re.match(r'\s*- .+?s - loss\: (\d.*?) - accuracy\: (\d.*?)'
                             ' - val_loss: (\d.*?) - val_accuracy\: (\d.*)', line)
            if match:
                # trainLossStr, trainAccStr = match.groups()
                # row = [float(trainLossStr), float(trainAccStr)]
                epochNum += 1
                row = [epochNum] + [float(valStr) for valStr in match.groups()]
                if len(row) != len(fieldNames):
                    raise Exception('Value count mismatch (%s)' % line)
                table.append(row)
                preciseCorrection = False
                continue

            # Parsing "Epoch 1: loss 0.001033556, acc 0.00056, val. loss 0.0010439, val. acc 0.0000000, last 1 epochs: 59.2020 s"
            match = re.match(r'.*Epoch (\d.*?)\: loss (\d.*?), acc.*? (\d.*?)'
                             ', val. loss (\d.*?), val. acc (\d.*?)[, ].*', line.strip('\r\n') + ' ')
            if match:
                epochNum = int(match.groups()[0])
                row = [epochNum] + [float(valStr) for valStr in match.groups()[1:]]
                if len(row) != len(fieldNames):
                    raise Exception('Value count mismatch (%s)' % line)
                table.append(row)
                preciseCorrection = False
            else:
                # Or parsing "18350.938 (209262.00) Epoch 39: loss 0.001005059, acc 0.00131, last 1 epochs: 286.9563 s"
                match = re.match(r'.* Epoch (\d.*?)\: loss (\d.*?), acc (\d.*?)[, ].*', line)
                if match:
                    epochNumStr, trainLossStr, trainAccStr = match.groups()
                    newEpochNum = int(epochNumStr)
                    if newEpochNum < epochNum:
                        raise Exception('Decreasing epoch number (%s)' % line)
                    epochNum = newEpochNum
                    trainLoss = float(trainLossStr)
                    trainAcc = float(trainAccStr)
                    if preciseCorrection:
                        if trainLoss != table[-1][1] or trainAcc != table[-1][2]:
                            raise Exception('Mismatch repeating correction (%s)' % line)
                    table[-1] = [epochNum, float(trainLossStr), float(trainAccStr)] + table[-1][3:]
                    preciseCorrection = True

    return pd.DataFrame(table, columns=fieldNames)

# RegExpStr is e.g. '.*\sLoss (\d.*?) \(.*Acc@1\s*(\d.*?)\s.*'.
# Return number column (1, 2, 3, ...) and found values
def parseLogByRegExp(srcFilePath, regExpStr):
    table = []
    with open(srcFilePath, 'r') as file:
        epochNum = 0
        for line in file:
            match = re.match(regExpStr, line)
            if match:
                epochNum += 1
                row = [epochNum] + [float(valStr) for valStr in match.groups()]
                # if len(row) != len(fieldNames):
                #     raise Exception('Value count mismatch (%s)' % line)
                table.append(row)

    return pd.DataFrame(table) # , columns=fieldNames)


# # Cuts diagram on the right if it exits from minY-maxY
# def cutDiag(xs, ys, minY, maxY):
#     for i in range(len(xs)):
#         if ys[i] < minY or ys[i] > maxY:


def analyzeColumn(plt, logTable, fieldName):
    if logTable.shape[0] <= 0:
        raise Exception('No log data to display')
    # xs = np.arange(1, logTable.shape[0] + 1)
    xs = logTable['epochNum']
    ys = logTable[fieldName]
    alpha = 0.5 if fieldName.find('val') < 0 else 0.2
    plt.plot(xs, ys, label=fieldName,
             marker='.', markersize=3, alpha=alpha)
    yDelta = ys.max() - ys.min()

    polyCoeffs = np.polyfit(xs, ys, 3)
    print(polyCoeffs)
    projXs = xs[0] + np.arange(0, (xs[len(xs) - 1] - xs[0]) * 2 + len(xs) * 3) / 2
    polyYs = np.polyval(polyCoeffs, projXs)
    # cutProj = cutDiag(projXs, polyYs, ys.min() - yDelta * 3, ys.max() + yDelta / 2)
    ax = plt.plot(projXs, polyYs, label=fieldName + 'Proj_3',
             linestyle='--', alpha=alpha + 0.2)

    polyCoeffs = np.polyfit(xs, ys, 4)
    print(polyCoeffs)
    polyYs = np.polyval(polyCoeffs, projXs)
    ax = plt.plot(projXs, polyYs, label=fieldName + 'Proj_4',
             linestyle=':', alpha=alpha + 0.2)

    # ax[0].autoscale(False)
    # ax[0].figure.bbox = {'left':xs[0], 'top': ys.min() - yDelta * 3, \
    #                      'width':projXs[-1], 'height':ys.max() + yDelta / 2}
    return (xs[0], ys.min() - yDelta * 3, projXs[-1], ys.max() + yDelta / 2)


def sigint_handler(*args):
    # global app
    global mainWindow

    """Handler for the SIGINT signal."""
    sys.stderr.write('Exiting\n')
    # app.exiting = True
    mainWindow.exiting = True
    QtGui.QApplication.quit()

if __name__ == '__main__':
      # t = parseLogByRegExp('progress.log', '.*\sLoss (\d.*?) \(.*Acc@1\s*(\d.*?)\s.*')
    # # filePath = 'log'
    # if len(sys.argv) > 1:
    #     filePath = sys.argv[1]
    #     logTable = getLogAsTable(filePath)
    #     # print(logTable)
    #
    #     ext = analyzeColumn(logTable, 'trainLoss')
    #     analyzeColumn(logTable, 'valLoss')
    #     plt.ylim([ext[1], ext[3]])
    #     plt.legend()
    #     plt.show()
    # else:
        app = QApplication(sys.argv)
        signal.signal(signal.SIGINT, sigint_handler)
        mainWindow = LogAnalysisQtMainWindow()
        mainWindow.setWindowState(Qt.WindowMaximized)
        mainWindow.show()
        # mainWindow.initUi()
        mainWindow.activateWindow()
        # mainWindow.selectFile(os.path.split(sys.argv[0])[0] + '/log')
        if len(sys.argv) > 1:
            mainWindow.selectFile(sys.argv[1])
        else:
            mainWindow.selectFile(r'OldLogs_Vis_ImagesNetTraining\AllLogsCopy\10_1_SpatDropout_Train_TestAcc0.45_log')
        sys.exit(app.exec_())
