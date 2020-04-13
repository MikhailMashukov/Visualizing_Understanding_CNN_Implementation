from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import DeepOptions

# More advanced AlexNet4 - with fractional max pools
class ImageModel3_Deeper(nn.Module):
    def __init__(self, num_classes=1000):
        super(ImageModel3_Deeper, self).__init__()

        mult = DeepOptions.netSizeMult
        towerCount = DeepOptions.towerCount
        self.towerCount = towerCount
        self.namedLayers = {'conv_1': nn.Conv2d(3, mult * 6, kernel_size=7, stride=2)}   # Input - 227 * 227
        self.namedLayers['conv_3'] = nn.Conv2d(mult * 16, mult * 24, 3, padding=0)
        for towerInd in range(towerCount):
                self.namedLayers['conv_2_%d' % (towerInd + 1)] = nn.Conv2d(mult * 6 // towerCount,  mult * 16 // towerCount, 3, padding=1)
                    # Input image 55 * 55, output - 77 * 77
                self.namedLayers['conv_22_%d' % (towerInd + 1)] = nn.Conv2d(mult * 16 // towerCount,  mult * 16 // towerCount, 3)
                    # Output image -
                # self.namedLayers['conv_3%d' % (towerInd + 1)] = nn.Conv2d(mult * 12 // towerCount, mult * 24 // towerCount, 3, padding=1)
                self.namedLayers['conv_4_%d' % (towerInd + 1)] = \
                        nn.Conv2d(mult * 24 // towerCount,  mult * 32 // towerCount, 3, padding=1)
                    # Input image 29 * 29
                self.namedLayers['conv_5_%d' % (towerInd + 1)] = \
                        nn.Conv2d(mult * 32 // towerCount,  mult * 24 // towerCount, 3, padding=1)
                    # Output image 15 * 15
#                 self.namedLayers['conv_6_%d' % (towerInd + 1)] = \
#                         nn.Conv2d(mult * 32 // towerCount,  mult * 16 // towerCount, 3, padding=1)
#                     # Output image 11 * 11

        self.subnet1 = nn.Sequential(OrderedDict([
                ('conv_1', self.namedLayers['conv_1']),
                ('relu_1', nn.ReLU(inplace=True)),
                ('max_pool_1', nn.FractionalMaxPool2d(3, output_ratio=0.67))]))
#                 ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2))]))
        subnets2 = []
        for towerInd in range(towerCount):
            name = 'conv_2_%d' % (towerInd + 1)
            subnets2.append(nn.Sequential(OrderedDict([
                    (name, self.namedLayers[name]),
                    ('relu_2_%d' % (towerInd + 1), nn.ReLU(inplace=True)),
                    ('max_pool_4_%d' % (towerInd + 1), nn.FractionalMaxPool2d(3, output_ratio=0.67)),
                    ('conv_22_%d' % (towerInd + 1), self.namedLayers['conv_22_%d' % (towerInd + 1)]),])))
        self.subnets2 = nn.ModuleList(subnets2)
        self.subnet3 = nn.Sequential(OrderedDict([
                ('relu_2', nn.ReLU(inplace=True)),
#                 ('max_pool_2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('max_pool_2', nn.FractionalMaxPool2d(3, output_ratio=0.67)),
                ('conv_3', self.namedLayers['conv_3']),
                ('relu_3_2', nn.ReLU(inplace=True))]))
#                 ('pad_3', nn.ZeroPad2d(1))]))
        subnets4 = []
        for towerInd in range(towerCount):
            subnets4.append(nn.Sequential(OrderedDict([
                    ('conv_4_%d' % (towerInd + 1), self.namedLayers['conv_4_%d' % (towerInd + 1)]),
                    ('relu_4_%d' % (towerInd + 1), nn.ReLU(inplace=True)),
                    ('max_pool_4_%d' % (towerInd + 1), nn.FractionalMaxPool2d(3, output_ratio=0.67)),
                #                     ('pad_5_%d' % (towerInd + 1), nn.ZeroPad2d(1)),
                    ('conv_5_%d' % (towerInd + 1), self.namedLayers['conv_5_%d' % (towerInd + 1)])])))
#                     ('relu_5_%d' % (towerInd + 1), nn.ReLU(inplace=True)),
#                     ('max_pool_5', nn.FractionalMaxPool2d(3, output_ratio=0.7)),
# #                     ('pad_6_%d' % (towerInd + 1), nn.ZeroPad2d(1)),
#                     ('conv_6_%d' % (towerInd + 1), self.namedLayers['conv_6_%d' % (towerInd + 1)])])))
        self.subnets4 = nn.ModuleList(subnets4)
        self.subnet5 = nn.Sequential(OrderedDict([
                ('relu_6_1', nn.ReLU(inplace=True)),
                ('max_pool_6', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ('avg_pool', nn.AdaptiveAvgPool2d((5, 5)))]))
        # self.net = nn.ModuleList([self.subnet1] + self.subnets2 + [self.subnet3] + \
        #         self.subnets4 + [self.subnet5])
        self.subnetList = [[self.subnet1], self.subnets2, [self.subnet3], \
                self.subnets4, [self.subnet5]]

        denseSize = 4096 // 16 * mult
        self.namedLayers['dense_1'] = nn.Linear(in_features=(mult * 24 * 5 * 5), out_features=denseSize)
        self.namedLayers['dense_2'] = nn.Linear(in_features=denseSize, out_features=denseSize)
        self.namedLayers['dense_3'] = nn.Linear(in_features=denseSize, out_features=num_classes)
        self.classifier = nn.Sequential(OrderedDict([
                ('dropout_1', nn.Dropout(p=0.5)),
                ('dense_1', self.namedLayers['dense_1']),
                ('relu_c_1', nn.ReLU(inplace=True)),
                ('dropout_1', nn.Dropout(p=0.5)),
                ('dense_2', self.namedLayers['dense_2']),
                ('relu_c_2', nn.ReLU(inplace=True)),
                ('dense_3', self.namedLayers['dense_3'])]))

    def forward(self, x):
        x = self.subnet1(x)
        xs = list(torch.split(x, x.shape[1] // self.towerCount, 1))
#         print(len(xs), xs[0].shape)
        for towerInd in range(self.towerCount):
            xs[towerInd] = self.subnets2[towerInd](xs[towerInd])
        x = torch.cat(xs, 1)
        x = self.subnet3(x)
        xs = list(torch.split(x, x.shape[1] // self.towerCount, 1))
        for towerInd in range(self.towerCount):
            xs[towerInd] = self.subnets4[towerInd](xs[towerInd])
        x = torch.cat(xs, 1)
        x = self.subnet5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def getLayer(self, layerName):
        return self.namedLayers[layerName]

    def getAllLayers(self):
        return self.namedLayers

    def saveState(self, fileName,
                  additInfo={}, additFileName=None):
        if 1:
            state = {'model': self.state_dict()}
            if additFileName is None:
                state.update(additInfo)
                torch.save(state, fileName)
                    # os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epochNum + 1))
            else:
                torch.save(state, fileName)
                torch.save(additInfo, additFileName)
        else:
            torch.save(self.state_dict(), fileName)

    def loadState(self, fileName):
        state = torch.load(fileName)
        # print('state ', state)
        savedStateDict = state['model']
        try:
            c_replacements = [['module.', ''], ]
            stateDict = {}
            for name, data in savedStateDict.items():
                for replRule in c_replacements:
                    name = name.replace(replRule[0], replRule[1])
                stateDict[name] = data
            result = self.load_state_dict(stateDict, strict=1)
            print('State loaded from %s (%s)' % (fileName, result))
            del state['model']
            return state
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

        if len(savedStateDict) != len(self.state_dict()):
            raise Exception('You are trying to load parameters for %d layers, but the model has %d' %
                            (len(savedStateDict), len(self.state_dict())))

        del state['model']
        return state

    def loadStateAdditInfo(self, fileName):
        return torch.load(fileName)


    class CutVersion(nn.Module):
        def __init__(self, baseModule, highestLayerName, allowCombinedLayers=False):
            super().__init__()
            self.baseModule = baseModule
            self.highestLayerName = highestLayerName
            # highestLayer = baseModule.getLayer(highestLayerName)
            self.allowCombinedLayers = allowCombinedLayers

        def forward(self, x):
            printShapes = 0 # True
            if printShapes:
                print(x.shape)
            towerCount = self.baseModule.towerCount
            combineLayer = self.allowCombinedLayers and \
                    not self.highestLayerName in self.baseModule.getAllLayers()
            for subnet in self.baseModule.subnetList:
                print('subnet len', len(subnet), 'shape', x.shape)
                foundTowerInd = self.findBlockInSubnet(subnet, self.highestLayerName)
                if len(subnet) == 1:
                    if foundTowerInd < 0:
                        if printShapes:
                            for name, module in subnet[0].named_children():
                                x = module(x)
                                print('After %s: %s' % (name, str(x.shape)))
                        else:
                            x = subnet[0](x)
                    else:
                        for name, module in subnet[0].named_children():
                            x = module(x)
                            if printShapes:
                                print('After %s: %s' % (name, str(x.shape)))
                            if self._isMatchedBlock(name, self.highestLayerName):
                                return x
                else:
                    xs = list(torch.split(x, x.shape[1] // towerCount, 1))

                    if foundTowerInd < 0:
                        for towerInd in range(towerCount):
                            if printShapes and towerInd == 0:
                                for name, module in subnet[towerInd].named_children():
                                    xs[towerInd] = module(xs[towerInd])
                                    print('After %s: %s' % (name, str(xs[towerInd].shape)))
                            else:
                                xs[towerInd] = subnet[towerInd](xs[towerInd])
                    else:
                        xsToReturn = []
                        for towerInd in ([foundTowerInd] if not combineLayer else range(len(subnet))):
                            for name, module in subnet[towerInd].named_children():
                                xs[towerInd] = module(xs[towerInd])
                                if self._isMatchedBlock(name, self.highestLayerName):
                                    xsToReturn.append(xs[towerInd])
                                    break
                        print('xsToReturn', len(xsToReturn))
                        return torch.cat(xsToReturn, 1)

                    x = torch.cat(xs, 1)
            print(x.shape)
            x = torch.flatten(x, 1)
            print(x.shape)
            for name, module in self.baseModule.classifier.named_children():
                x = module(x)
                print('After %s: %s' % (name, str(x.shape)))
                if name == self.highestLayerName:
                    return x
            raise Exception('Layer %s not found' % self.highestLayerName)

        # Returns index in subnet's list or < 0
        def findBlockInSubnet(self, subnet, blockToFindName):
            for towerInd, module in enumerate(subnet):
                for name, layer in module.named_children():
                    if self._isMatchedBlock(name, blockToFindName):
                        return towerInd
            return -1

        def _isMatchedBlock(self, blockName, blockToFindName):
            return blockName == blockToFindName or \
                    (self.allowCombinedLayers and blockName[ : len(blockToFindName) + 1] == blockToFindName + '_')


# Multiple towers with some connections between them
class ImageModel4_ConnectedTowers(nn.Module):
    def __init__(self, num_classes=1000):
        super(ImageModel4_ConnectedTowers, self).__init__()

        mult = DeepOptions.netSizeMult
        towerCount = DeepOptions.towerCount
        self.towerCount = towerCount
        self.namedLayers = {'conv_1': nn.Conv2d(3, mult * 6, kernel_size=11, stride=4, padding=2)}   # Input - 227 * 227
        # self.namedLayers['conv_3'] = nn.Conv2d(mult * 16, mult * 24, 3, padding=1)

        self.subnet1 = nn.Sequential(OrderedDict([
                ('conv_1', self.namedLayers['conv_1']),
                ('relu_1', nn.ReLU(inplace=True)),
                ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2))]))
        subnets2 = []
        for towerInd in range(towerCount):
            name = 'conv_2_%d' % (towerInd + 1)
            self.namedLayers['conv_2_%d' % (towerInd + 1)] = \
                    nn.Conv2d(mult * 6 // towerCount,  mult * 16 // towerCount, 5, padding=2)
                # Input image 27 * 27
            subnets2.append(nn.Sequential(OrderedDict([
                    (name, self.namedLayers[name]),
                    ('relu_3_1', nn.ReLU(inplace=True)),
                    ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2))])))
        self.subnets2 = nn.ModuleList(subnets2)
        self.subnetUnite2 = nn.Sequential(OrderedDict([
                ('conv_u_2', nn.Conv2d(mult * 16,  mult, 1))]))

        subnets3 = []
        for towerInd in range(towerCount):
            self.namedLayers['conv_3_%d' % (towerInd + 1)] = \
                    nn.Conv2d(mult * 16 // towerCount + mult,  mult * 24 // towerCount, 3, padding=1)
            subnets3.append(nn.Sequential(OrderedDict([
                    ('conv_3_%d' % (towerInd + 1), self.namedLayers['conv_3_%d' % (towerInd + 1)]),
                    ('relu_3_%d' % (towerInd + 1), nn.ReLU(inplace=True))])))
        self.subnets3 = nn.ModuleList(subnets3)
        self.subnetUnite3 = nn.Sequential(OrderedDict([
                ('conv_u_3', nn.Conv2d(mult * 24,  mult, 1))]))

        subnets4 = []
        for towerInd in range(towerCount):
            self.namedLayers['conv_4_%d' % (towerInd + 1)] = \
                    nn.Conv2d(mult * 24 // towerCount + mult,  mult * 24 // towerCount, 3, padding=1)
                # Input image 13 * 13
            self.namedLayers['conv_5_%d' % (towerInd + 1)] = \
                    nn.Conv2d(mult * 24 // towerCount,  mult * 16 // towerCount, 3, padding=1)
            subnets4.append(nn.Sequential(OrderedDict([
                    ('conv_4_%d' % (towerInd + 1), self.namedLayers['conv_4_%d' % (towerInd + 1)]),
                    ('relu_4_%d' % (towerInd + 1), nn.ReLU(inplace=True)),
                    ('conv_5_%d' % (towerInd + 1), self.namedLayers['conv_5_%d' % (towerInd + 1)])])))
        self.subnets4 = nn.ModuleList(subnets4)
        self.subnet5 = nn.Sequential(OrderedDict([
                ('relu_5_1', nn.ReLU(inplace=True)),
                ('max_pool_5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('avg_pool', nn.AdaptiveAvgPool2d((6, 6)))]))
        # self.net = nn.ModuleList([self.subnet1] + self.subnets2 + [self.subnet3] + \
        #         self.subnets4 + [self.subnet5])
        self.subnetList = [[self.subnet1], self.subnets2, [self.subnetUnite2],
                self.subnets3, [self.subnetUnite3], \
                self.subnets4, [self.subnet5]]

        denseSize = 2048 // 16 * mult
        self.namedLayers['dense_1'] = nn.Linear(in_features=(mult * 16 * 6 * 6), out_features=denseSize)
        self.namedLayers['dense_2'] = nn.Linear(in_features=denseSize, out_features=denseSize)
        self.namedLayers['dense_3'] = nn.Linear(in_features=denseSize, out_features=num_classes)
        self.namedLayers['dropout_1'] = nn.Dropout(p=0.5)
        self.namedLayers['relu_c_1'] = nn.ReLU(inplace=True)
        self.namedLayers['dropout_2'] = nn.Dropout(p=0.5)
        self.namedLayers['relu_c_2'] = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(OrderedDict([
                (name, self.namedLayers[name]) for name in ['dropout_1', 'dense_1', 'relu_c_1',
                    'dropout_2', 'dense_2', 'relu_c_2', 'dense_3']]))

    def forward(self, x):
        x = self.subnet1(x)
        xs = list(torch.split(x, x.shape[1] // self.towerCount, 1))
#         print(len(xs), xs[0].shape)
        for towerInd in range(self.towerCount):
            xs[towerInd] = self.subnets2[towerInd](xs[towerInd])
        x = torch.cat(xs, 1)

        convU2x = self.subnetUnite2(x)
        # xs = list(torch.split(x, x.shape[1] // self.towerCount, 1))
        for towerInd in range(self.towerCount):
            xs[towerInd] = self.subnets3[towerInd](torch.cat([xs[towerInd], convU2x], 1))
        x = torch.cat(xs, 1)
        convU3x = self.subnetUnite3(x)
        for towerInd in range(self.towerCount):
            xs[towerInd] = self.subnets4[towerInd](torch.cat([xs[towerInd], convU3x], 1))
        x = torch.cat(xs, 1)
        x = self.subnet5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def getLayer(self, layerName):
        return self.namedLayers[layerName]

    def getAllLayers(self):
        return self.namedLayers

    def saveState(self, fileName,
                  additInfo={}, additFileName=None):
        if 1:
            state = {'model': self.state_dict()}
            if additFileName is None:
                state.update(additInfo)
                torch.save(state, fileName)
                    # os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epochNum + 1))
            else:
                torch.save(state, fileName)
                torch.save(additInfo, additFileName)
        else:
            torch.save(self.state_dict(), fileName)

    def loadState(self, fileName):
        state = torch.load(fileName)
        # print('state ', state)
        savedStateDict = state['model']
        try:
            c_replacements = [['module.', ''], ]
            stateDict = {}
            for name, data in savedStateDict.items():
                for replRule in c_replacements:
                    name = name.replace(replRule[0], replRule[1])
                stateDict[name] = data
            result = self.load_state_dict(stateDict, strict=1)
            print('State loaded from %s (%s)' % (fileName, result))
            del state['model']
            return state
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

        if len(savedStateDict) != len(self.state_dict()):
            raise Exception('You are trying to load parameters for %d layers, but the model has %d' %
                            (len(savedStateDict), len(self.state_dict())))

        del state['model']
        return state

    def loadStateAdditInfo(self, fileName):
        return torch.load(fileName)


    class CutVersion(nn.Module):
        def __init__(self, baseModule, highestLayerName, allowCombinedLayers=False):
            super().__init__()
            self.baseModule = baseModule
            self.highestLayerName = highestLayerName
            # highestLayer = baseModule.getLayer(highestLayerName)
            self.allowCombinedLayers = allowCombinedLayers

        def forward(self, x):
            printShapes = 0 # True
            if printShapes:
                print(x.shape)
            towerCount = self.baseModule.towerCount
            combineLayer = self.allowCombinedLayers and \
                    not self.highestLayerName in self.baseModule.getAllLayers()
            for subnet in self.baseModule.subnetList:
                print('subnet len', len(subnet), 'shape', x.shape)
                foundTowerInd = self.findBlockInSubnet(subnet, self.highestLayerName)
                if len(subnet) == 1:
                    if foundTowerInd < 0:
                        if printShapes:
                            for name, module in subnet[0].named_children():
                                x = module(x)
                                print('After %s: %s' % (name, str(x.shape)))
                        else:
                            x = subnet[0](x)
                    else:
                        for name, module in subnet[0].named_children():
                            x = module(x)
                            if printShapes:
                                print('After %s: %s' % (name, str(x.shape)))
                            if self._isMatchedBlock(name, self.highestLayerName):
                                return x
                else:
                    xs = list(torch.split(x, x.shape[1] // towerCount, 1))

                    if foundTowerInd < 0:
                        for towerInd in range(towerCount):
                            if printShapes and towerInd == 0:
                                for name, module in subnet[towerInd].named_children():
                                    xs[towerInd] = module(xs[towerInd])
                                    print('After %s: %s' % (name, str(xs[towerInd].shape)))
                            else:
                                xs[towerInd] = subnet[towerInd](xs[towerInd])
                    else:
                        xsToReturn = []
                        for towerInd in ([foundTowerInd] if not combineLayer else range(len(subnet))):
                            for name, module in subnet[towerInd].named_children():
                                xs[towerInd] = module(xs[towerInd])
                                if self._isMatchedBlock(name, self.highestLayerName):
                                    xsToReturn.append(xs[towerInd])
                                    break
                        print('xsToReturn', len(xsToReturn))
                        return torch.cat(xsToReturn, 1)

                    x = torch.cat(xs, 1)
            print(x.shape)
            x = torch.flatten(x, 1)
            print(x.shape)
            for name, module in self.baseModule.classifier.named_children():
                x = module(x)
                print('After %s: %s' % (name, str(x.shape)))
                if name == self.highestLayerName:
                    return x
            raise Exception('Layer %s not found' % self.highestLayerName)

        # Returns index in subnet's list or < 0
        def findBlockInSubnet(self, subnet, blockToFindName):
            for towerInd, module in enumerate(subnet):
                for name, layer in module.named_children():
                    if self._isMatchedBlock(name, blockToFindName):
                        return towerInd
            return -1

        def _isMatchedBlock(self, blockName, blockToFindName):
            return blockName == blockToFindName or \
                    (self.allowCombinedLayers and blockName[ : len(blockToFindName) + 1] == blockToFindName + '_')

