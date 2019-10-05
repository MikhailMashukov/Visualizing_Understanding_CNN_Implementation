# Should be run with C:\Program Files\Python35

# import copy
# # import math
import numpy as np
# # import random
# import sys
# # import time

class CDataCache:
    def __init__(self, maxMemoryMB=256):
        self.maxMemory = maxMemoryMB * (1 << 20)
        self.maxLruSize = max(maxMemoryMB, 8)

        self.data = dict()
        self.usedMemory = 0
        self.lrus = [set(), set()]    # New and previous sets of cache item names

    def getObject(self, name):              # Returns the object or None
        cacheItem = self.data.get(name)
        if not cacheItem is None:
            if len(self.lrus[0]) >= self.maxLruSize:
                self.lrus = [set(), self.lrus[0]]
            self.lrus[0].add(name)
        return cacheItem

    def saveObject(self, name, value):      # Adds or replaces the object in the cache
        prevValue = self.data.get(name)
        if not prevValue is None:
            self.usedMemory -= self.getApproxObjectSize(prevValue)

        if self.usedMemory > self.maxMemory:
            self.partialClean()
        self.data[name] = value
        self.usedMemory += self.getApproxObjectSize(value)

    def getUsedMemory(self):
        return self.usedMemory

    def clear(self):
        self.data = dict()
        self.usedMemory = 0

    def partialClean(self):
        newDataDict = dict()
        newUsedMemory = 0
        for name, value in self.data.items():
            if name in self.lrus[0] or name in self.lrus[1]:
                newDataDict[name] = value
                newUsedMemory += self.getApproxObjectSize(value)
        self.data = newDataDict
        self.usedMemory = newUsedMemory

        if self.usedMemory > self.maxMemory / 2:
            self.clear()

    def saveState_OpenedFile(self, file):
        import pickle

        pickle.dump(self.data, file)
        pickle.dump(self.usedMemory, file)

    def loadState_OpenedFile(self, file):
        import pickle

        self.data = pickle.load(file)
        self.usedMemory = pickle.load(file)

    @staticmethod
    def getApproxObjectSize(value):   # Approximately
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, list):
            if value:
                return CDataCache.getApproxObjectSize(value[0]) * len(value)
            else:
                return 16
        elif isinstance(value, str):
            return 16 + len(value)
        else:
            try:
                return value.numpy().nbytes     # ResourceVariable
            except:
                return 32