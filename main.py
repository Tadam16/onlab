import torchvision
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Upsample
from torch.nn import ReLU
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
    def __init__(self, middlechannels = (35, 70, 140, 280, 560), inchannels = 9, outchannels = 2):
        super().__init__()

        self.pool = MaxPool2d(2)
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True) #todo biztos igy?
        self.depth = len(middlechannels)

        self.blocks = []
        for i in range(self.depth):
            level = []
            for j in range(i + 1):

                inputs = j * middlechannels[i - j]
                if j == 0 and i == 0:
                    inputs += inchannels
                elif j == 0:
                    inputs += middlechannels[i - 1]
                else:
                    inputs += middlechannels[i - j + 1]

                outputs = middlechannels[i - j]
                if j == self.depth - 1 and i == self.depth-1:
                    outputs = outchannels

                level.append(Block(inputs, outputs))
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

def loss(output, expected):



def train():
    model = NestedUnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=1e-8)

if __name__ == '__main__':
    for i in range(1, 5):
        print(i)