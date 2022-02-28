from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
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
        self.relu1 = ReLU()
        self.conv2 = Conv2d(inChannels, outChannels, 3)
        self.switchnorm2 = SwitchNorm2d() #todo params
        self.relu2 = ReLU()

    def forward(self, x):
        return self.relu2(self.switchnorm2(self.conv2(self.relu1(self.switchnorm1(self.conv1(x))))))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
