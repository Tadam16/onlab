import numpy as np
import torch.utils.data
import os
import torch
import torchvision.models
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
import glob
from torchvision import transforms



class Vessel12Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.mask = None
        self.img = None
        self.imgList = glob.glob("vessel12dataset\imgs\*.mhd")
        self.maskList = glob.glob("vessel12dataset\masks\*.mhd")

    def loadimage(self, idx):
        self.img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgList[idx]))
        self.img = (np.clip(self.img, -1200, 600) + 1200) / 1800.0
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(self.maskList[idx]))

    def __getitem__(self, idx):
        idx = idx + 4

        resimg = self.img[idx - 4:idx + 5, :, :]
        resmask = self.mask[idx, :, :]

        return torch.from_numpy(resimg).float(), torch.from_numpy(resmask).long()

    def __len__(self):
        if self.img is None:
            return 0
        return self.img.shape[0] - 8
