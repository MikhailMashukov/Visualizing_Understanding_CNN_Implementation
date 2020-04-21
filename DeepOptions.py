import math

fastDebugMode = 0      # True means that net is very small and heavy statistics is not gathered
towerCount = 4
if fastDebugMode:
    modelClass = 'DebugImageModel4'
    netSizeMult = 1
else:
    netSizeMult = 64 * 2   # Default for wide ResNet - 64 * 2
    # modelClass = 'AlexnetModel4'
    # modelClass = 'ImageModel4'
    # modelClass = 'ChanConvModel'
    # modelClass = 'ChanUnitingModel'
    # modelClass = 'ChanMatrixModel'
    modelClass = 'MyWideResNet'
# additLayerCounts = (1, 1)         # For ImageModel4 and similar, max (2, 2)
additLayerCounts = [2, 3, 3, 3, 2]     # For ResNets, actually number of blocks. 2, 3, 3, 2 - most probably 32 layers
# additLayerCounts = [1, 2, 2, 1]   # Most probably20 layers
# deeperNet = 0          # True means that exended network is created but smaller subnet can be loaded from checkpoint
# hlAndIndicsDivision = False   # To divide input data onto 3 blocks. Currently implemented only for GRU network
# minDivisionSize = 10000      # Minimum for some convolution operation estimation after which it is divided onto two independent blocks
# useBatchNorm = 0       # False means simple linear all-to-all layers
# predictPriceChanges = True    # False means old max buy-sell and sell-buy profits
# useTargetClasses = False      # True means prediction of profit class: [0, 1, 0, 0, 0, 0, 0] (False - 2 max profit values)
# studyType = 2          # Deep = 2, DeepDiff = 3, GruNet - 4
# if not useBatchNorm:
#     convDropoutKeepProb = 1    # 1 - turn off
#     dropoutKeepProb = 0.65      # 1 - turn off
# else:
#     convDropoutKeepProb = 1
#     dropoutKeepProb = 1
# dropoutByChannels = False
# noiseChangeCount = 3

if 0:
    imagesMainFolder = 'ImageNetPart'
    classCount = 24
else:
    imagesMainFolder = 'ImageNet'
    if fastDebugMode:
        classCount = 110
    else:
        classCount = 1000

# startSeed = None
#
# batchSize = 64
# if fastDebugMode:
#     batchSize = 8
#     startCandleIndex = 0
#     trainSetSize = 100
#     additCandlesToLoad = 5000
#     testSetSize = 150
#     testSet2Size = testSetSize
# else:
#     startCandleIndex = 200000
#     if forexData:
#         trainSetSize = 120000
#         additCandlesToLoad = 50000
#         testSetSize = 5000
#         testSet2Size = 1000
#     else:
#         trainSetSize = 37000
#         additCandlesToLoad = 50000
#         testSetSize = 7000
#         testSet2Size = 500
# testSetSizes = [testSetSize, testSetSize, testSet2Size]
