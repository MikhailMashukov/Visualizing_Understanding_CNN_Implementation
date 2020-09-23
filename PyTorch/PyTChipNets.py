import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

# For transforms' extension
from collections.abc import Sequence
from PIL import Image, ImageOps
import torchvision
from torchvision.transforms import functional as F

import numpy as np
import os

# import DeepOptions

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

# @torch.jit.unused
class PadTo(object):
    def __init__(self, targetSize, fill=0, padding_mode="constant"):
        super().__init__()
        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if not isinstance(targetSize, Sequence) or len(targetSize) not in [2]:
            raise ValueError("Target size must 2 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.targetSize = targetSize
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if isinstance(img, Image.Image):
            # img = np.array(img)
            shape = img.size
            # print('image shape', shape)
            padding = [self.targetSize[0] - shape[0], self.targetSize[1] - shape[1]]
        else:
            shape = img.shape
            # print('shape', shape) #, 'mode', img.mode)
            padding = [self.targetSize[1] - shape[0], self.targetSize[0] - shape[1]]
        # print('target shape', target.size, 'mode', target.mode)
        # print('param2', param2.items())

        if padding[0] <= 0 and padding[1] <= 0:
            return img
        padding = tuple((((v + 1) // 2) if v >= 0 else 0) for v in padding)

        if isinstance(img, Image.Image):
            # print('expanding image', img, padding)
            # print('expanded', ImageOps.expand(img, border=padding, fill=self.fill).size)
            return ImageOps.expand(img, border=padding, fill=self.fill)
        else:
            padding = tuple((v, v) for v in padding)
            # newTarget = param2.copy()
            # newTarget['masks'] =
            if len(shape) > 2:
                padding = (padding[0], padding[1], (0, 0))
            # print('expanding', padding)
            return np.pad(img, padding, constant_values=self.fill)
                # ImageOps.expand(img, border=padding, fill=self.fill), \

    def __repr__(self):
        return self.__class__.__name__ + '(targetSize={0}, fill={1}, padding_mode={2})'.\
            format(self.targetSize, self.fill, self.padding_mode)

class Crop(object):
    def __init__(self, top, left, height, width):
        self.rect = (top, left, height, width)

    def __call__(self, img):
        return F.crop(img, self.rect[1], self.rect[0], self.rect[3], self.rect[2])
            # left, top, width, height

class ToTensor_Chip(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        # print('target to t.')
        target = F.to_tensor(target)
        return image, target



class PennDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        print('%d images, %d masks' % (len(self.imgs), len(self.masks)))

    def __getitem__(self, idx):
        # print('get', idx)
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        # img = np.array(img)

        mask = np.array(mask)
        # print('loaded', img.shape, img.dtype, mask.shape, mask.dtype)
        # mask = np.array(mask, dtype=np.uint8)
        # print('converted')
        mask[mask > 0] = 255
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        # target = torchvision.transforms.ToPILImage()(mask)
        target = Image.fromarray(mask)

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            # print('transform', img.__class__.__name__)
            img = self.transforms(img)
            target = self.transforms(target)
            # print(target.max())


        return img, target # {'mask': target}

    def __len__(self):
        return len(self.imgs)

class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = ['2D 2.bmp']
        self.masks = ['2D_2_mask.png']
        print(self.imgs)
        print('%d images, %d masks' % (len(self.imgs), len(self.masks)))

    def __getitem__(self, idx):
        # print('get', idx)
        img_path = os.path.join(self.root, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # mask[mask > 0] = 255
        target = Image.fromarray(mask)

        if self.transforms is not None:
            # print('transform', img.__class__.__name__)
            img = self.transforms(img)
            target = self.transforms(target)
            # print(target.max())

        return img, target

    def __len__(self):
        return len(self.imgs)


class UnetConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            UnetConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = UnetConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = UnetConv(in_channels, out_channels)


    def forward(self, x1, x2):    # x1 - smaller
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # print('up', x1.shape, x2.shape, diffY, diffX)

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class ChipNet(nn.Module):
    def __init__(self, num_classes=2, basePlaneCount=16, norm_layer=None):
        super().__init__()
        planeCount = basePlaneCount
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.namedLayers = {}
        bilinear = True

        self.conv1 = conv3x3(3, planeCount, 1)
        self.bn1 = norm_layer(planeCount)
        # self.conv12 = conv1x1(3, planeCount, 1)
        # self.bn12 = norm_layer(planeCount)
        self.relu = nn.ReLU(inplace=True)

        factor = 2 if bilinear else 1
        self.down1 = Down(planeCount, basePlaneCount * 2)
        self.down2 = Down(basePlaneCount * 2, basePlaneCount * 4)
        self.up3 = Up(basePlaneCount * 12 // factor, basePlaneCount * 2, bilinear)
        self.up4 = Up(basePlaneCount * 6 // factor, planeCount, bilinear)

        self.conv2 = conv3x3(planeCount, planeCount)
        self.bn2 = norm_layer(planeCount)
        self.conv3 = conv3x3(planeCount, planeCount)
        self.bn3 = norm_layer(planeCount)
        self.conv4 = conv1x1(planeCount, num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # print('forward', len(x), x)
        # x = x[0]
        # x2 = self.conv12(x)
        # x2 = self.bn12(x2)
        # x2 = self.relu(x2)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # x = self.maxpool(x)

        x2 = self.down1(x1)
        # print('2', x2.shape)
        x3 = self.down2(x2)
        # print('3', x3.shape)

        x = self.conv2(x1)
        # print('22', x.shape)
        x = self.bn2(x)
        x1 = self.relu(x)

        x = self.up3(x3, x2)
        # print('4', x.shape)
        x = self.up4(x, x1)
        # print('4', x.shape)
        # x = self.conv3(x)   # * x2
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.conv4(x)
        return x

def createSimpleChipNet(num_classes, trainable_backbone_layers=3, **kwargs):
    return ChipNet(num_classes)







# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnext50_32x4d', 'resnext101_32x8d']
#
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }
#
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width) #, groups=groups)
#         self.bn1 = norm_layer(width)
#         print('conv2: w %d, stride %d, groups %d, dilation %d' % \
#                 (width, stride, groups, dilation))
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion) #, groups=groups)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#     # HighestLayer here can be some of self.<layer> or 'start' or 'final_conv'
#     def forward_CutModel(self, x, highestLayer):
#         identity = x
#
#         out = self.conv1(x)
#         if highestLayer == self.conv1:
#             return out
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         if highestLayer == self.conv2:
#             return out
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         if highestLayer == self.conv3:
#             return out
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         if highestLayer == 'start':
#             return identity
#
#         out += identity
#         if highestLayer == 'final_conv':
#             return out
#         out = self.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(ResNet, self).__init__()
#         self.setHighestLayer(None)
#         self.layerDepths = layers
#
#         self.namedLayers = {}
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.namedLayers['conv_1'] = self.conv1
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 2, 64, layers[0])
#         self.layer2 = self._make_layer(block, 3, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 4, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 5, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.namedLayers['avg_pool_5'] = self.avgpool
#         spatAnSubnetWidth = DeepOptions.netSizeMult // 8
#         self.spatAnSubnet = self._makeSpatialAnalysisSubnet(block, 9, spatAnSubnetWidth, 4)
#
#         self.fc = nn.Linear(512 * block.expansion + spatAnSubnetWidth, num_classes)
#         self.namedLayers['dense_1'] = self.fc
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, layerNum, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         convLayerNamePrefix = 'conv_%d' % layerNum
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         # print('layers.append', layers[-1])
#         self.inplanes = planes * block.expansion
#         print(convLayerNamePrefix, 'block:', self.inplanes, planes, self.base_width)
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups, # if i != 2 else 1,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#         for i, layerBlock in enumerate(layers):
#             # Block of the Bottleneck type is implied here
#             self.namedLayers[convLayerNamePrefix + '%d1' % (i + 1)] = layerBlock.conv1
#             self.namedLayers[convLayerNamePrefix + '%d2' % (i + 1)] = layerBlock.conv2   # Will be considered main
#             self.namedLayers[convLayerNamePrefix + '%d3' % (i + 1)] = layerBlock.conv3
#
#         return nn.Sequential(*layers)
#
#     # def _makeSpatialAnalysisSubnet(self, sizeBeforeAvgPool, sizeMult):
#     def _makeSpatialAnalysisSubnet(self, block, layerNum, width, blocks):
#         # placeConv1 = nn.Conv2d(1, sizeMult, kernel_size=7, stride=2, padding=3,
#         #                        bias=False)
#         layers = []
#         convLayerNamePrefix = 'conv_%d' % layerNum
#         conv1 = conv1x1(self.inplanes, width)   # TODO: another conv -> wider
#         layers.append(conv1)
#         self.namedLayers[convLayerNamePrefix + '1'] = conv1
#
#         for i in range(blocks):
#             layers.append(block(width, width // 4, groups=2,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=self._norm_layer))
#         for i, layerBlock in enumerate(layers[1:]):
#             # Block of the Bottleneck type is implied here
#             self.namedLayers[convLayerNamePrefix + '%d1' % (i + 2)] = layerBlock.conv1
#             self.namedLayers[convLayerNamePrefix + '%d2' % (i + 2)] = layerBlock.conv2   # Will be considered main
#             self.namedLayers[convLayerNamePrefix + '%d3' % (i + 2)] = layerBlock.conv3
#
# #         print('out chans', layers[-1].conv3.out_channels, width)
# #         fc = nn.Linear(layers[-1].conv3.out_channels, width)
# #         layers.append(fc)
# #         self.namedLayers['dense_2'] = fc
#
#         # The same self.avgpool will be used here
#         return nn.Sequential(*layers)
#
#
#     # Takes layer name like 'conv_11' or None
#     def setHighestLayer(self, highestLayerName):
#         self.highestLayerName = highestLayerName
#         if self.highestLayerName is None:
#             self.forward = self._forward_impl
#         else:
#             self.forward = self._forward_Impl_CutModel
#
#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         xBA = self.layer4(x)   # X before average pool
#
# #         print('xBA', xBA.shape)
#         x2 = self.spatAnSubnet(xBA)
# #         x2 = xBA
# #         for name, l in self.spatAnSubnet.named_children():
# #             x2 = l(x2)
# #             print('x2', name, x2.shape)
#
#         x = torch.cat([xBA, x2], 1)
# #         print('concatenated', x.shape)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#
#         x = self.fc(x)
#         return x
#
#     def _forward_Impl_CutModel(self, x):
#         if self.highestLayerName in self.namedLayers:
#             highestLayer = self.namedLayers[self.highestLayerName]
#         else:
#             highestLayer = None
#         (layerNum, blockNum, innerLayerSuffix) = self.parseLayerName(self.highestLayerName)
#
#         x = self.conv1(x)
#         if highestLayer == self.conv1:
#             return x
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         if layerNum == 2:
#             return self._forward_layer_CutModel(self.layer1, blockNum, innerLayerSuffix, x)
#         x = self.layer1(x)
#         if layerNum == 3:
#             return self._forward_layer_CutModel(self.layer2, blockNum, innerLayerSuffix, x)
#         x = self.layer2(x)
#         if layerNum == 4:
#             return self._forward_layer_CutModel(self.layer3, blockNum, innerLayerSuffix, x)
#         x = self.layer3(x)
#         if layerNum == 5:
#             return self._forward_layer_CutModel(self.layer4, blockNum, innerLayerSuffix, x)
#         xBA = self.layer4(x)
#
#         if layerNum == 9:
#             return self._forward_layer_CutModel(self.spatAnSubnet, blockNum, innerLayerSuffix, xBA)
#         x2 = self.spatAnSubnet(xBA)
#         x = torch.cat([xBA, x2], 1)
#
#         x = self.avgpool(x)
#         if highestLayer == self.avgpool:
#             return x
#         x = torch.flatten(x, 1)
#
#         x = self.fc(x)
#         if highestLayer == self.fc:
#             return x
#         raise Exception('Layer %s not found' % self.highestLayerName)
#
#     # def forward(self, x):
#     #     return self._forward_impl(x)
#
#     @staticmethod
#     def parseLayerName(layerName):
#         if layerName.find('conv_') == 0:
#             layerBlockSubstr = layerName[len('conv_') : len('conv_') + 2]
#             # layerNum = int(layerName[len('conv_')])
#             # blockNum = int(layerName[len('conv_N')])
#             innerLayerSuffix = layerName[len('conv_NM') : ]
#         elif layerName.find('start_') == 0:
#             innerLayerSuffix = 'start'
#             layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 3]
#         elif layerName.find('final_conv_') == 0:
#             innerLayerSuffix = 'final_conv'
#             layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 3]
#         elif layerName.find('dense_') == 0:
#             innerLayerSuffix = 'dense'
#             layerBlockSubstr = layerName[len(innerLayerSuffix) + 1 : len(innerLayerSuffix) + 2]
#         else:
#             raise Exception('Unexpected layer name %s' % layerName)
#
#         layerNum = int(layerBlockSubstr[0])
#         if len(layerBlockSubstr) >= 2:
#             blockNum = int(layerBlockSubstr[1])
#         else:
#             blockNum = None
#         return (layerNum, blockNum, innerLayerSuffix)
#
#     def _forward_layer_CutModel(self, layerModule, blockNum, innerLayerSuffix, x):
#         # if self.highestLayerName.find('conv_') == 0:
#         #     hishestLayerNameSuffix = self.highestLayerName[len('conv_N') : ]
#         #     hishestBlockInd = int(hishestLayerNameSuffix[0]) - 1
#         #     innerHighestLayer = self.namedLayers[self.highestLayerName]
#         # elif self.highestLayerName.find('start_') == 0:
#
#         # print('highest ', self.highestLayerName, blockNum, innerLayerSuffix)
#         for blockInd, layerBlock in enumerate(layerModule):
#             # layerBlock = layerModule[blockInd]   # This is block from layers.append(block(...)) in _make_layer
#             if blockInd + 1 < blockNum:
#                 x = layerBlock(x)
#             else:
#                 # Block of the Bottleneck type is implied here
#                 if self.highestLayerName in self.namedLayers:
#                     return layerBlock.forward_CutModel(x, self.namedLayers[self.highestLayerName])
#                 else:
#                     return layerBlock.forward_CutModel(x, innerLayerSuffix)
#
#
#     def getLayer(self, layerName):
#         return self.namedLayers[layerName]
#
#     def getAllLayers(self):
#         return self.namedLayers
#
#     def saveState(self, fileName,
#                   additInfo={}, additFileName=None):
#         if 1:
#             state = {'model': self.state_dict()}
#             if additFileName is None:
#                 state.update(additInfo)
#                 torch.save(state, fileName)
#                     # os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epochNum + 1))
#             else:
#                 torch.save(state, fileName)
#                 torch.save(additInfo, additFileName)
#         else:
#             torch.save(self.state_dict(), fileName)
#
#     def loadState(self, fileName):
#         state = torch.load(fileName)
#         # print('state ', state)
#         savedStateDict = state['model']
#         try:
#             c_replacements = [['module.', ''], ]
#             stateDict = {}
#             for name, data in savedStateDict.items():
#                 for replRule in c_replacements:
#                     name = name.replace(replRule[0], replRule[1])
#                 stateDict[name] = data
#             result = self.load_state_dict(stateDict, strict=1)
#             print('State loaded from %s (%s)' % (fileName, result))
#             del state['model']
#             return state
#         except Exception as ex:
#             print("Error in loadState: %s" % str(ex))
#
#         if len(savedStateDict) != len(self.state_dict()):
#             raise Exception('You are trying to load parameters for %d layers, but the model has %d' %
#                             (len(savedStateDict), len(self.state_dict())))
#
#         del state['model']
#         return state
#
#     def loadStateAdditInfo(self, fileName):
#         return torch.load(fileName)
#
#
#     # class CSourceBlockCalculator:
#     #     @staticmethod
#     def get_source_block_calc_func(self, layerName):
#         return self._get_source_block_calc_impl(layerName,
#                 ResNet, [2] * len(self.layerDepths))
#
#     def _get_source_block_calc_impl(self, layerName, thisClass, afterLayerStrides):
#             if layerName[:6] == 'dense_':
#                 return thisClass.get_entire_image_block
#
#             size = 7
#             if layerName == 'conv_1':
#                 def get_source_block(x, y):
#                     source_xy_0 = (x * 2 - 3, y * 2 - 3)
#                     return thisClass.correctZeroCoords(source_xy_0, size)
#
#                 return get_source_block
#             size += 2 * 2
#             if layerName in ['max_pool_1', 'conv_211']:
#                 def get_source_block(x, y):
#                     source_xy_0 = (x * 4 - 5, y * 4 - 5)
#                     return thisClass.correctZeroCoords(source_xy_0, size)
#
#                 return get_source_block
#
#             (highestLayerNum, highestBlockNum, innerLayerSuffix) = self.parseLayerName(layerName)
#             stride = 4
#             leftUpShift = 5
#             found = False
#             for layerInd, layerDepth in enumerate(self.layerDepths):
#                 for blockNum in range(1, layerDepth + 1):
#                     # curLayerNamePrefix = 'conv_%d%d' % (layerInd + 2, blockNum)
#
#                     if not (layerInd + 2 == highestLayerNum and blockNum == highestBlockNum):
#                         size += stride * 2
#                         leftUpShift += stride   # Padding is usually 1
#                     else:
#                         print('Matching layer %d, %d, %s' % (highestLayerNum, highestBlockNum, innerLayerSuffix))
#                         if innerLayerSuffix == '1' or innerLayerSuffix == 'start':
#                             if blockNum == 1:
#                                 stride //= 2
#                         else:
#                             size += stride * 2
#                             leftUpShift += stride   # Padding is usually 1
#                             if innerLayerSuffix == '2':
#                                 pass
#                             else:
#                                 size += stride * 2
#                                 leftUpShift += stride
#                                 if innerLayerSuffix in ['3', 'final_conv']:
#                                     found = True
#                                 else:
#                                     raise Exception('Unexpected layer name %s' % self.highestLayerName)
#
#                         def get_source_block(x, y):
#                             source_xy_0 = (x * stride - leftUpShift, y * stride - leftUpShift)
#                             return thisClass.correctZeroCoords(source_xy_0, size)
#
#                         return get_source_block
#                 stride *= 1.5 if layerInd <= 3 else 2
#
#
#             return None
#
#     @staticmethod
#     def correctZeroCoords(source_xy_0, size):
#         return (0 if source_xy_0[0] < 0 else int(source_xy_0[0]),
#                 0 if source_xy_0[1] < 0 else int(source_xy_0[1]),
#                 int(source_xy_0[0] + size), int(source_xy_0[1] + size))
#
#     @staticmethod
#     def get_entire_image_block(x, y):
#         return (0, 0, 224, 224)
#
#
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)
#
#
# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def my_resnet(pretrained=False, progress=True, **kwargs):
#     r"""Added by me by analogy. Combination of ResNeXt and wide ResNet ideas
#     """
#     kwargs['groups'] = DeepOptions.towerCount
#     kwargs['width_per_group'] = DeepOptions.netSizeMult // DeepOptions.towerCount
#     return _resnet('wide_resnet', Bottleneck, DeepOptions.additLayerCounts,
#                    pretrained, progress, **kwargs)
#


if __name__ == '__main__':
    import transforms as T
    from train import *
    import utils

    def get_transform(train):
        transforms = []
        targetSize = None   # (640, 512)
        # targetSize = (32, 32)
        transforms.append(Crop(250, 300, 704, 512))
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            # transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
            transforms.append(torchvision.transforms.RandomVerticalFlip())
        if not targetSize is None:
            transforms.append(PadTo(targetSize, fill=0))
            transforms.append(torchvision.transforms.CenterCrop((targetSize[1], targetSize[0])))
        transforms.append(torchvision.transforms.ToTensor())
        # return T.Compose(transforms)
        return torchvision.transforms.Compose(transforms)

    dataset = ChipDataset(r'E:\Projects\Freelance\INIRSibir\Images', get_transform(train=True))
    dataset_test = ChipDataset(r'E:\Projects\Freelance\INIRSibir\Images', get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    if len(indices) > 50:
        testImageCount = 50
    else:
        testImageCount = len(indices) // 3

    if testImageCount > 0:
        dataset = torch.utils.data.Subset(dataset, indices[:-testImageCount])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-testImageCount:])
    else:
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset_test = torch.utils.data.Subset(dataset_test, indices)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, num_workers=4,  # shuffle=True,
        sampler=train_sampler, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=4,  # shuffle=False,
        sampler=test_sampler, collate_fn=utils.collate_fn)

    num_classes = 1
    model = createSimpleChipNet(num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,   # 0.005
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,  # 3
                                                   gamma=0.5)
    img, target = dataset[0]

    num_epochs = 500

    i = 1
    t = 1
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        model.eval()
        with torch.no_grad():
            if len(img.shape) == 3:
                img.unsqueeze_(0)
            prediction = model(img.to(device))
            prediction = prediction[0].cpu().numpy().transpose(1, 2, 0)
            # prediction[prediction > 1] = 1
            # prediction[prediction < 0] = 0
            print(prediction.shape, prediction.dtype, prediction.min(), np.mean(prediction), prediction.max())
        # fig = plt.imshow(np.squeeze(prediction, 2),
        #            vmin=-1, vmax=2, cmap='rainbow');
        # plt.colorbar()
        # plt.show()

     # def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
     # device, epoch, print_freq):
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, \
                        device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device, num_classes)
        # print(x)