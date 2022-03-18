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
from dataset import Vessel12Dataset

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.switchnorm1 = SwitchNorm2d(outChannels) #todo params
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
        self.switchnorm2 = SwitchNorm2d(outChannels) #todo params

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
        results = [[]]

        for i in range(self.depth):
            levelresults = []
            results.append(levelresults)
            for j in range(i + 1):

                block = self.blocks[i][j]

                if(i == 0 and j == 0):
                    input = x
                elif(j == 0):
                    input = self.pool(results[i-1][0])
                else:
                    input = self.up(results[i][j-1])

                for k in range(j):
                    input = torch.cat([input, results[i - k - 1][j - k - 1]], dim=1)

                levelresults.append(block(input))

        return results[self.depth - 1][self.depth - 1]

def loss(output, expected):
    print("loss function not implemented")


def train():
    model = NestedUnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=1e-8)

if __name__ == '__main__':
    model = NestedUnet()
    dataset = Vessel12Dataset()
    dataset.loadimage(5)
    (img, mask) = dataset.__getitem__(67)
    output = model(img)
    for i in range(1, 5):
        print(i)