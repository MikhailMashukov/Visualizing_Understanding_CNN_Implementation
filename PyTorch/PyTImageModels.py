import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import DeepOptions

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# AlexNet from torchvision.examples. First advance in the middle of the 1st epoch,
# top-1 train/test accuracy after 5 epochs - 23.3%/30.3%
class AlexNet_TV(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_TV, self).__init__()

        mult = DeepOptions.netSizeMult
        self.net = nn.Sequential(
            nn.Conv2d(3, mult * 4, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(mult * 4, mult * 12, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(mult * 12, mult * 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mult * 24, mult * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mult * 16, mult * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        denseSize = 4096 // 16 * mult
        self.namedLayers = {'conv_1': list(self.net.children())[0]}
        self.namedLayers['dense_1'] = nn.Linear(in_features=(mult * 16 * 6 * 6), out_features=denseSize)
        self.namedLayers['dense_2'] = nn.Linear(in_features=denseSize, out_features=denseSize)
        self.namedLayers['dense_3'] = nn.Linear(in_features=denseSize, out_features=num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.namedLayers['dense_1'],
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.namedLayers['dense_2'],
            nn.ReLU(inplace=True),
            self.namedLayers['dense_3'],
        )

    def forward(self, x):
        x = self.net(x)
        x = self.avgpool(x)
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
            c_replacements = [['module.', ''],
                              ['net.0.', 'net.conv_1.'], ['net.4.', 'net.conv_2.'],
                              ['net.8.', 'net.conv_3.'], ['net.10.', 'net.conv_4.'],
                              ['net.12.', 'net.conv_5.']]
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


def alexnet_TV(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet_TV(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
