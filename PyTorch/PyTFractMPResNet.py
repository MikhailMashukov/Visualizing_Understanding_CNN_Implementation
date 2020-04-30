import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import DeepOptions
from . import PyTResNets

class FractMaxPoolResNet(PyTResNets.ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FractMaxPoolResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer)
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
            replace_stride_with_dilation = [False, False, False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.namedLayers['conv_1'] = self.conv1                                   # Output images - 112 * 112
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=0.66) #, padding=1)
        self.layer1 = self._make_layer(block, 2, 64, layers[0])                   # With fract. m. p. - 73 * 73
        self.maxpool2 = nn.FractionalMaxPool2d(kernel_size=3, output_size=36)     # 24 * 24
        self.layer2 = self._make_layer(block, 3, 96, layers[1], stride=1,         # 37 * 37
                                       dilate=replace_stride_with_dilation[0])
        self.maxpool3 = nn.FractionalMaxPool2d(kernel_size=3, output_size=24)     # 24 * 24
        self.layer3 = self._make_layer(block, 4, 128, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.maxpool4 = nn.FractionalMaxPool2d(kernel_size=3, output_size=16)
        self.layer4 = self._make_layer(block, 5, 224, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.maxpool5 = nn.FractionalMaxPool2d(kernel_size=3, output_size=10)
        self.layer5 = self._make_layer(block, 6, 320, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[3])
        self.layer6 = self._make_layer(block, 7, 512, layers[5], stride=2,
                                       dilate=replace_stride_with_dilation[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                               # Input images - 6 * 6
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

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.maxpool4(x)
        x = self.layer4(x)
        x = self.maxpool5(x)
        x = self.layer5(x)
        x = self.layer6(x)

        # print('avgp:', x.shape)
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
        x = self.maxpool2(x)
        if layerNum == 3:
            return self._forward_layer_CutModel(self.layer2, blockNum, innerLayerSuffix, x)
        x = self.layer2(x)
        x = self.maxpool3(x)
        if layerNum == 4:
            return self._forward_layer_CutModel(self.layer3, blockNum, innerLayerSuffix, x)
        x = self.layer3(x)
        x = self.maxpool4(x)
        if layerNum == 5:
            return self._forward_layer_CutModel(self.layer4, blockNum, innerLayerSuffix, x)
        x = self.layer4(x)
        x = self.maxpool5(x)
        if layerNum == 6:
            return self._forward_layer_CutModel(self.layer5, blockNum, innerLayerSuffix, x)
        x = self.layer5(x)
        if layerNum == 7:
            return self._forward_layer_CutModel(self.layer6, blockNum, innerLayerSuffix, x)
        x = self.layer6(x)
        x = self.avgpool(x)
        if highestLayer == self.avgpool:
            return x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if highestLayer == self.fc:
            return x
        raise Exception('Layer %s not found' % self.highestLayerName)


    def get_source_block_calc_func(self, layerName):
        return self._get_source_block_calc_impl(layerName, 
                FractMaxPoolResNet, [1.5] * 4 + [2])


def fract_max_pool_resnet(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = DeepOptions.towerCount
    kwargs['width_per_group'] = DeepOptions.netSizeMult // DeepOptions.towerCount
    assert pretrained == False
    return FractMaxPoolResNet(PyTResNets.Bottleneck, DeepOptions.additLayerCounts,
            **kwargs)
