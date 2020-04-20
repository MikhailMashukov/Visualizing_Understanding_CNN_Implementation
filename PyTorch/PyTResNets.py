import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import DeepOptions

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # print('conv2: w %d, stride %d, groups %d, dilation %d' % \
        #         (width, stride, groups, dilation))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    # HighestLayer here can be some of self.<layer> or 'start' or 'final_conv'
    def forward_CutModel(self, x, highestLayer):
        identity = x

        out = self.conv1(x)
        if highestLayer == self.conv1:
            return out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if highestLayer == self.conv2:
            return out
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if highestLayer == self.conv3:
            return out
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if highestLayer == 'start':
            return identity

        out += identity
        if highestLayer == 'final_conv':
            return out
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.setHighestLayer(None)
        self.layerDepths = layers

        self.namedLayers = {}
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.namedLayers['conv_1'] = self.conv1
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 2, 64, layers[0])
        self.layer2 = self._make_layer(block, 3, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 5, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.namedLayers['avg_pool_5'] = self.avgpool
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.namedLayers['dense_1'] = self.fc

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, layerNum, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        convLayerNamePrefix = 'conv_%d' % layerNum
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # print('layers.append', layers[-1])
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        for i, layerBlock in enumerate(layers):
            # Block of the Bottleneck type is implied here
            self.namedLayers[convLayerNamePrefix + '%d1' % (i + 1)] = layerBlock.conv1
            self.namedLayers[convLayerNamePrefix + '%d2' % (i + 1)] = layerBlock.conv2   # Will be considered main
            self.namedLayers[convLayerNamePrefix + '%d3' % (i + 1)] = layerBlock.conv3

        return nn.Sequential(*layers)

    # Takes layer name like 'conv_11' or None
    def setHighestLayer(self, highestLayerName):
        self.highestLayerName = highestLayerName
        if self.highestLayerName is None:
            self.forward = self._forward_impl
        else:
            self.forward = self._forward_Impl_CutModel

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _forward_Impl_CutModel(self, x):
        if self.highestLayerName in self.namedLayers:
            highestLayer = self.namedLayers[self.highestLayerName]
        else:
            highestLayer = None
        (layerNum, blockNum, innerLayerSuffix) = self.parseLayerName(self.highestLayerName)

        x = self.conv1(x)
        if highestLayer == self.conv1:
            return x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if layerNum == 2:
            return self._forward_layer_CutModel(self.layer1, blockNum, innerLayerSuffix, x)
        x = self.layer1(x)
        if layerNum == 3:
            return self._forward_layer_CutModel(self.layer2, blockNum, innerLayerSuffix, x)
        x = self.layer2(x)
        if layerNum == 4:
            return self._forward_layer_CutModel(self.layer3, blockNum, innerLayerSuffix, x)
        x = self.layer3(x)
        if layerNum == 5:
            return self._forward_layer_CutModel(self.layer4, blockNum, innerLayerSuffix, x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if highestLayer == self.avgpool:
            return x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if highestLayer == self.fc:
            return x
        raise Exception('Layer %s not found' % self.highestLayerName)

    # def forward(self, x):
    #     return self._forward_impl(x)

    @staticmethod
    def parseLayerName(layerName):
        if layerName.find('conv_') == 0:
            layerBlockSubstr = layerName[len('conv_') : len('conv_') + 2]
            # layerNum = int(layerName[len('conv_')])
            # blockNum = int(layerName[len('conv_N')])
            innerLayerSuffix = layerName[len('conv_NM') : ]
        elif layerName.find('start_') == 0:
            innerLayerSuffix = 'start'
            layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 3]
        elif layerName.find('final_conv_') == 0:
            innerLayerSuffix = 'final_conv'
            layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 3]
        elif layerName.find('dense_') == 0:
            innerLayerSuffix = 'dense'
            layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 2]
        else:
            raise Exception('Unexpected layer name %s' % layerName)

        layerNum = int(layerBlockSubstr[0])
        if len(layerBlockSubstr) >= 2:
            blockNum = int(layerBlockSubstr[1])
        else:
            blockNum = None
        return (layerNum, blockNum, innerLayerSuffix)

    def _forward_layer_CutModel(self, layerModule, blockNum, innerLayerSuffix, x):
        # if self.highestLayerName.find('conv_') == 0:
        #     hishestLayerNameSuffix = self.highestLayerName[len('conv_N') : ]
        #     hishestBlockInd = int(hishestLayerNameSuffix[0]) - 1
        #     innerHighestLayer = self.namedLayers[self.highestLayerName]
        # elif self.highestLayerName.find('start_') == 0:

        # print('highest ', self.highestLayerName, blockNum, innerLayerSuffix)
        for blockInd, layerBlock in enumerate(layerModule):
            # layerBlock = layerModule[blockInd]   # This is block from layers.append(block(...)) in _make_layer
            if blockInd + 1 < blockNum:
                x = layerBlock(x)
            else:
                # Block of the Bottleneck type is implied here
                if self.highestLayerName in self.namedLayers:
                    return layerBlock.forward_CutModel(x, self.namedLayers[self.highestLayerName])
                else:
                    return layerBlock.forward_CutModel(x, innerLayerSuffix)


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


    # class CSourceBlockCalculator:
    #     @staticmethod
    def get_source_block_calc_func(self, layerName):
            thisClass = ResNet  # .CSourceBlockCalculator
            if layerName[:6] == 'dense_':
                return thisClass.get_entire_image_block

            size = 7
            if layerName == 'conv_1':
                def get_source_block(x, y):
                    source_xy_0 = (x * 2 - 3, y * 2 - 3)
                    return thisClass.correctZeroCoords(source_xy_0, size)

                return get_source_block
            size += 2 * 2
            if layerName in ['max_pool_1', 'conv_211']:
                def get_source_block(x, y):
                    source_xy_0 = (x * 4 - 5, y * 4 - 5)
                    return thisClass.correctZeroCoords(source_xy_0, size)

                return get_source_block

            (highestLayerNum, highestBlockNum, innerLayerSuffix) = self.parseLayerName(layerName)
            stride = 4
            leftUpShift = 5
            found = False
            for layerInd, layerDepth in enumerate(self.layerDepths):
                for blockNum in range(1, layerDepth + 1):
                    # curLayerNamePrefix = 'conv_%d%d' % (layerInd + 2, blockNum)

                    if not (layerInd + 2 == highestLayerNum and blockNum == highestBlockNum):
                        size += stride * 2
                        leftUpShift += stride   # Padding is usually 1
                    else:
                        print('Matching layer %d, %d, %s' % (highestLayerNum, highestBlockNum, innerLayerSuffix))
                        if innerLayerSuffix == '1' or innerLayerSuffix == 'start':
                            if blockNum == 1:
                                stride //= 2
                        else:
                            size += stride * 2
                            leftUpShift += stride   # Padding is usually 1
                            if innerLayerSuffix == '2':
                                pass
                            else:
                                size += stride * 2
                                leftUpShift += stride
                                if innerLayerSuffix in ['3', 'final_conv']:
                                    found = True
                                else:
                                    raise Exception('Unexpected layer name %s' % self.highestLayerName)

                        def get_source_block(x, y):
                            source_xy_0 = (x * stride - leftUpShift, y * stride - leftUpShift)
                            return thisClass.correctZeroCoords(source_xy_0, size)

                        return get_source_block
                stride *= 2

            # size += 4 * 2
            # if layerName == 'conv_21':
            #     def get_source_block(x, y):
            #         source_xy_0 = ((x - 1) * 4 - 5, (y - 1) * 4 - 5)
            #         return thisClass.correctZeroCoords(source_xy_0, size)
            #
            #     return get_source_block
            # size += 4 * 2
            # if layerName == 'conv_213':
            #     def get_source_block(x, y):
            #         source_xy_0 = ((x - 2) * 4 - 5, (y - 2) * 4 - 5)
            #         return thisClass.correctZeroCoords(source_xy_0, size)
            #
            #     return get_source_block
            # size += 16 * 2
            # if layerName == 'conv_5':
            #     def get_source_block(x, y):
            #         source_xy_0 = ((x - 1) * 4 - 5, (y - 1) * 4 - 5)
            #         return thisClass.correctZeroCoords(source_xy_0, size)
            #
            #     return get_source_block

            return None

    @staticmethod
    def correctZeroCoords(source_xy_0, size):
        return (0 if source_xy_0[0] < 0 else source_xy_0[0],
                0 if source_xy_0[1] < 0 else source_xy_0[1],
                source_xy_0[0] + int(size), source_xy_0[1] + int(size))

    @staticmethod
    def get_entire_image_block(x, y):
        return (0, 0, 224, 224)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def my_wide_resnet(pretrained=False, progress=True, **kwargs):
    r"""Added by me by analogy. Combination of ResNeXt and wide ResNet ideas
    """
    kwargs['groups'] = DeepOptions.towerCount
    kwargs['width_per_group'] = DeepOptions.netSizeMult // DeepOptions.towerCount
    return _resnet('wide_resnet', Bottleneck, DeepOptions.additLayerCounts,
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)