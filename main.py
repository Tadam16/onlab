import torchvision
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Upsample
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from switchnorm import SwitchNorm2d


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.switchnorm1 = SwitchNorm2d() #todo params
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
        self.switchnorm2 = SwitchNorm2d() #todo params

    def forward(self, x):
        return self.relu(self.switchnorm2(self.conv2(self.relu(self.switchnorm1(self.conv1(x))))))

class NestedUnet(Module):
    def __init__(self, channels = (9, 35, 70, 120, 280, 560)):
        super().__init__()

        self.pool = MaxPool2d(2)
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True) #todo biztos igy?
        self.depth = len(channels) - 1

        self.blocks = []
        for i in range(1, self.depth + 1):
            level = []
            for j in range(1, i + 1):
                level.append(Block(channels[i - 1], channels[i]))
            self.blocks.append(level)

    def forward(self, x):
        results = [[x]]
        for i in range(1, self.depth + 1):

            if i != 1:
                levelresults = self.pool(results[i - 1][0])
            else:
                levelresults = [results[i - 1][0]]

            for j in range(1, i + 1):

                if j == 1:
                    input = levelresults[j - 1]
                else:
                    input = self.up(levelresults[j - 1])

                for k in range(1, i):
                    levelinput = self.crop(results[k][j], input)
                    input = torch.cat([input, levelinput], dim=1)

                levelresults.append(self.blocks[i-1][j-1](input))

            results.append(levelresults)

        return results[self.depth][self.depth]

    def crop(self, encFeatures, x):
        (_,_,H,W) = x.shape
        return torchvision.transforms.CenterCrop([H,W])(encFeatures)

def loss():


def train():


if __name__ == '__main__':
    for i in range(1, 5):
        print(i)