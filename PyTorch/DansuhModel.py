"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 1e-4   # SGD with 0.01 doesn't work for them
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use
# modify this to point to your data directory
# INPUT_ROOT_DIR = '/root/Visualiz_Zeiler/ImageNet'         # 'alexnet_data_in'
TRAIN_IMG_DIR =  '/root/Visualiz_Zeiler/ImageNet/train'   # 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'PyTLogs'
LOG_DIR = OUTPUT_DIR + '/TbLogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoints'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        if 0:         # For loading old models with not named layers
          self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))  # (b x 256 x 6 x 6)
          self.namedLayers = {'conv_1': list(self.net.children())[0]}
        else:
          self.namedLayers = {
            'conv_1': nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            'conv_2': nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            'conv_3': nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            'conv_4': nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            'conv_5': nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
          }
          self.net = nn.Sequential(OrderedDict([       # OrderedDict doesn't help, need custom naming
            ('conv_1', self.namedLayers['conv_1']),
            ('relu_1', nn.ReLU()),
            ('norm_1', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),  # section 3.3
            ('max_pool_1', nn.MaxPool2d(kernel_size=3, stride=2)),  # (b x 96 x 27 x 27)
            ('conv_2', self.namedLayers['conv_2']),
            ('relu_2', nn.ReLU()),
            ('norm_2', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),
            ('max_pool_2', nn.MaxPool2d(kernel_size=3, stride=2)),  # (b x 256 x 13 x 13)
            ('conv_3', self.namedLayers['conv_3']),
            ('relu_3', nn.ReLU()),
            ('conv_4', self.namedLayers['conv_4']),
            ('relu_4', nn.ReLU()),
            ('conv_5', self.namedLayers['conv_5']),
            ('relu_5', nn.ReLU()),
            ('max_pool_5', nn.MaxPool2d(kernel_size=3, stride=2)),  # (b x 256 x 6 x 6)
          ]))
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


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


def printProgress(str):
    with open(OUTPUT_DIR + '/progress.log', 'a') as file:
        file.write(str + '\n')

if __name__ == '__main__':
    trainAlexNet()

def trainAlexNet(learnRate=LR_INIT, printWeightStats=False):
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    # print('AlexNet created')

    # create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    # print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=learnRate)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    # print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    total_steps = 1
    blockNum = 0
    valLossInfo = 'val. loss 0, val. acc 0'
    for epochNum in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 50 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.6f} \tAcc: {}'
                        .format(epochNum + 1, total_steps, loss.item(), accuracy.item()))
                    blockNum += 1
                    printProgress('Epoch %d: loss %.7g, acc %.6f, %s ' \
                                  '(actual epoch: %d)' %
                                  (blockNum, loss.item(), accuracy.item(), valLossInfo,
                                   epochNum + 1))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # print out gradient values and parameter average values
            if total_steps % 1000 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    if printWeightStats:
                        print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            if printWeightStats:
                                print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            if printWeightStats:
                                print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            if total_steps % ((64000 if epochNum == 0 else 256000) // BATCH_SIZE) == 0:
                additInfo = {'epoch': epochNum,
                             'total_steps': total_steps,
                             'seed': seed}
                alexnet.saveState(os.path.join(CHECKPOINT_DIR, 'PyTImWeights_Epoch{}.h5'.format(blockNum)),
                                  additInfo)
                        # os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epochNum + 1))

            total_steps += 1
