from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import Upsample
from torch.nn import ReLU
from torch.nn import Sigmoid
import torch

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, 1, (1, 1))
        #self.switchnorm1 = SwitchNorm2d(outChannels)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3, 1, (1, 1))
        #self.switchnorm2 = SwitchNorm2d(outChannels)
        #self.dropout = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
        #return self.relu(self.switchnorm2(self.dropout(self.conv2(self.relu(self.switchnorm1(self.conv1(x)))))))

class EndBlock(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, 1, (1, 1))
        #self.switchnorm1 = SwitchNorm2d(outChannels)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3, 1, (1, 1))
        #self.switchnorm2 = SwitchNorm2d(outChannels)
        #self.dropout = torch.nn.Dropout2d(0.2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        #return self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))

class NestedUnet(Module):
    def __init__(self, middlechannels = (35, 70, 140, 280, 560), inchannels = 10, outchannels = 1):
        super().__init__()

        self.pool = MaxPool2d(2)
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.depth = len(middlechannels)

        self.blocks = torch.nn.ModuleList()
        for i in range(self.depth):
            level = torch.nn.ModuleList()
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
                    level.append(EndBlock(inputs, outputs))
                else:
                    level.append(Block(inputs, outputs))

            self.blocks.append(level)

    def forward(self, x):
        results = []
        x = self.pool(x) #Lowering resolution to spare memory
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

        return self.up(results[self.depth - 1][self.depth - 1]) #resetting resolution