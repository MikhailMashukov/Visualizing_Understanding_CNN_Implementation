from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import DeepOptions

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# AlexNet, based on torchvision.examples. First advance in the middle of the 1st epoch,
# top-1 train/test accuracy after 5 epochs - 23.3%/30.3%.
# Now also added towers and changed number of channels
class AlexNet4(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet4, self).__init__()

        mult = DeepOptions.netSizeMult
        towerCount = DeepOptions.towerCount
        self.towerCount = towerCount
        self.namedLayers = {'conv_1': nn.Conv2d(3, mult * 6, kernel_size=11, stride=4, padding=2)}
        self.namedLayers['conv_3'] = nn.Conv2d(mult * 16, mult * 24, 3, padding=1)
        # self.namedLayers['conv_4'] = nn.Conv2d(mult * 24, mult * 16, 3, padding=1)
#         self.namedLayers['conv_5'] = nn.Conv2d(mult * 16, mult * 16, 3, padding=1)
        # if towerCount == 1:
        #     self.namedLayers['conv_2'] = nn.Conv2d(mult * 4,  mult * 12, 5, padding=2)
        # else:
        for towerInd in range(towerCount):
                self.namedLayers['conv_2_%d' % (towerInd + 1)] = nn.Conv2d(mult * 6 // towerCount,  mult * 16 // towerCount, 5, padding=2)
                # self.namedLayers['conv_3%d' % (towerInd + 1)] = nn.Conv2d(mult * 12 // towerCount, mult * 24 // towerCount, 3, padding=1)
                self.namedLayers['conv_4_%d' % (towerInd + 1)] = nn.Conv2d(mult * 24 // towerCount,  mult * 24 // towerCount, 3, padding=1)
                self.namedLayers['conv_5_%d' % (towerInd + 1)] = nn.Conv2d(mult * 24 // towerCount,  mult * 16 // towerCount, 3, padding=1)

        self.subnet1 = nn.Sequential(OrderedDict([
                ('conv_1', self.namedLayers['conv_1']),
                ('relu_1', nn.ReLU(inplace=True)),
                ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2))]))
        subnets2 = []
        for towerInd in range(towerCount):
            name = 'conv_2_%d' % (towerInd + 1)
            subnets2.append(nn.Sequential(OrderedDict([
                    (name, self.namedLayers[name])])))
        self.subnets2 = nn.ModuleList(subnets2)
        self.subnet3 = nn.Sequential(OrderedDict([
                ('relu_3_1', nn.ReLU(inplace=True)),
                ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv_3', self.namedLayers['conv_3']),
                ('relu_3_2', nn.ReLU(inplace=True))]))
        subnets4 = []
        for towerInd in range(towerCount):
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
        self.subnetList = [[self.subnet1], self.subnets2, [self.subnet3], \
                self.subnets4, [self.subnet5]]

        denseSize = 4096 // 16 * mult
        self.namedLayers['dense_1'] = nn.Linear(in_features=(mult * 16 * 6 * 6), out_features=denseSize)
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
                            (len(savedStateDict) != len(self.state_dict())))

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
            print(x.shape)
            towerCount = self.baseModule.towerCount
            combineLayer = self.allowCombinedLayers and \
                    not self.highestLayerName in self.baseModule.getAllLayers()
            for subnet in self.baseModule.subnetList:
                print('subn', len(subnet))
                foundTowerInd = self.findBlockInSubnet(subnet, self.highestLayerName)
                if len(subnet) == 1:
                    if foundTowerInd < 0:
                        x = subnet[0](x)
                    else:
                        for name, module in subnet[0].named_children():
                            x = module(x)
                            if self._isMatchedBlock(name, self.highestLayerName):
                                return x
                else:
                    xs = list(torch.split(x, x.shape[1] // towerCount, 1))

                    if foundTowerInd < 0:
                        for towerInd in range(towerCount):
                            xs[towerInd] = subnet[towerInd](xs[towerInd])
                    else:
                        xsToReturn = []
                        for towerInd in ([foundTowerInd] if not combineLayer else range(len(subnet))):
                            for name, module in subnet[towerInd].named_children():
                                print('n1', name)
                                xs[towerInd] = module(xs[towerInd])
                                if self._isMatchedBlock(name, self.highestLayerName):
                                    xsToReturn.append(xs[towerInd])
                                    break
                        print('xsToReturn', len(xsToReturn))
                        return torch.cat(xsToReturn, 1)

                    x = torch.cat(xs, 1)
            x = torch.flatten(x, 1)
            for name, module in self.baseModule.classifier.named_children():
                x = module(x)
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


def alexnet4(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet4(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
