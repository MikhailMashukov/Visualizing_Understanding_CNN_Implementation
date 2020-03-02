import copy
import datetime
# import pickle
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
# import os
# import random
# import sys
import _thread
import threading
# import time

from MyUtils import *


class DownloadThreadParams:
    def __init__(self):
        # threadInd
        # ids
        # threadName
        # downloadStats object reference
        # sqlConn
        self.writeLock = _thread.allocate_lock()

class DownloadStats:
    def __init__(self):
        self.updateLock = _thread.allocate_lock()
        # self.total = 0
        # self.loaded = 0

        self.finishedThreadCount = 0

def calc_MultiThreaded(threadCount, threadFunc, ids, options,
                          printMessTempl = 'Calculating activation tops for %d epoch(s)'):
    if not ids:
        return
    print(printMessTempl % len(ids))
    # sys.stdout.flush()
    params = DownloadThreadParams()
    # params.pageStep = threadCount
    params.downloadStats = DownloadStats()

    # Running working download and processing threads
    try:
        n = threading.active_count()
        if len(ids) < threadCount:
            threadCount = len(ids)
        for i in range(threadCount):
            params.threadInd = i
            params.threadName = 'Thread %d' % i
            if ids:
                l = len(ids)
                params.ids = [ids[j] for j in range(i, l, threadCount)]
            t = threading.Thread(None, threadFunc,
                                 params.threadName, [params, options])
            params = copy.copy(params)
            # Params for threads must be different but downloadStats in them - the same object

            t.daemon = True
            t.start()
            n = threading.active_count()
            time.sleep(0.3)
            # if hasattr(DownloadOptions, 'threadStartDelay'):
            #     time.sleep(DownloadOptions.threadStartDelay)

        while True:
            with params.downloadStats.updateLock:
                if params.downloadStats.finishedThreadCount >= threadCount:
                    break
            time.sleep(0.1)
    except Exception as errtxt:
        print(errtxt)


def attachCoordinates(data):
    shape = data.shape
    arr = np.arange(0, shape[0])
    grid = np.meshgrid(arr, np.arange(0, shape[1]))
        # np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
    coords = np.vstack([grid[0].flatten(), grid[1].flatten(), data.flatten()])
    return coords

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
    # shift = activations.shape[1] + channelMargin
    if fillValue is None:
        fillValue = 0 if activations.dtype in [np.uint8, np.uint32] else -1
    colMarginData = np.full([activations.shape[1], channelMargin] + list(activations.shape[3:]),
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
    if not fullList:
        return activations[0]
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


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    ''' 2D Pooling on last 2 dimensions
    https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    if 0:
        # Window view of A
        output_shape = A.shape[:-2] + ((A.shape[-2] - kernel_size)//stride + 1,
                (A.shape[-1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(A, shape=output_shape + kernel_size,
                         strides=(stride*A.strides[-2],          # Something else should be here
                                  stride*A.strides[-1]) + A.strides,
                         writeable=False)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        # aixs = (len(A.shape) - )
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif len(A.shape) > 2:
        output_shape = A.shape[:-2] + ((A.shape[-2] - kernel_size)//stride + 1,
                (A.shape[-1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A2 = A.reshape((-1, A.shape[-2], A.shape[-1]))
        A_ws = []
        for i in range(A2.shape[0]):
            A_w = as_strided(A2[i], shape=output_shape[-2:] + kernel_size,
                             strides=(stride*A2.strides[-2],
                                      stride*A2.strides[-1]) + A2.strides[-2:],
                             writeable=False)
            A_ws.append(A_w.reshape(-1, *kernel_size))
        A_w = np.stack(A_ws, axis=0)
        if pool_mode == 'max':
            return A_w.max(axis=(2,3)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(2,3)).reshape(output_shape)
    else:      # Source variant supporting only 2D-arrays
        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(A, shape=output_shape + kernel_size,
                         strides=(stride*A.strides[0],
                                  stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)