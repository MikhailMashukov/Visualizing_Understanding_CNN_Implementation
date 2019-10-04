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
        self.data = dict()
        self.usedMemory = 0

    def getObject(self, name):              # Returns the object or None
        return self.data.get(name)

    def saveObject(self, name, value):      # Adds or replaces the object in the cache
        prevValue = self.data.get(name)
        if not prevValue is None:
            self.usedMemory -= self.getApproxObjectSize(prevValue)

        if self.usedMemory > self.maxMemory:
            self.clear()
        self.data[name] = value
        self.usedMemory += self.getApproxObjectSize(value)

    def clear(self):
        self.data = dict()
        self.usedMemory = 0

    def getUsedMemory(self):
        return self.usedMemory


    @staticmethod
    def getApproxObjectSize(value):   # Approximately
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif value is list:
            if value:
                return CDataCache.getApproxObjectSize(value[0]) * len(value)
            else:
                return 16
        elif value is str:
            return 16 + len(value)
        else:
            return 32