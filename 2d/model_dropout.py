import numpy as np
import torch.utils.data
from torch.nn import functional as F
import torch

class Block(torch.nn.Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.norm = torch.nn.Dropout2d(0.2)
        self.conv1 = torch.nn.Conv2d(inChannels, outChannels, 3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(outChannels, outChannels, 3, 1, 1)

    def forward(self, x):
        return self.relu(self.conv2(self.norm(self.relu(self.conv1(x)))))


class Encoder(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.encblocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        blockoutputs = []

        for block in self.encblocks:
            x = block(x)
            blockoutputs.append(x)
            x = self.pool(x)

        return blockoutputs


class Decoder(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.upconvs = torch.nn.ModuleList(
            torch.nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1))
        self.decblocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) -1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) -1 ):
            x = self.upconvs[i](x)

            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim = 1)
            x = self.decblocks[i](x)

        return x

    def crop(self, encFeatures, x):
        (_,_,H,W) = x.shape
        return F.interpolate(encFeatures, (H, W))
        #return torchvision.transforms.CenterCrop([H,W])(encFeatures)



class Unet(torch.nn.Module):

    def __init__(self, encChannels = (1, 16, 32, 64), decChannels = (64, 32, 16), nbClasses = 1, outSize = (512, 512)):
        super().__init__()

        self.Encoder = Encoder(encChannels)
        self.Decoder = Decoder(decChannels)

        self.head = torch.nn.Conv2d(decChannels[-1], nbClasses, 1, 1)
        self.outsize = outSize
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x[::-1][0], x[::-1][1:])
        x = self.head(x)
        return self.sigmoid(x)
