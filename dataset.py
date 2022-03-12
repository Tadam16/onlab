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



class Vessel12Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.imgList = glob.glob("vessel12dataset\imgs\*.mhd")
        self.maskList = glob.glob("vessel12dataset\masks\*.mhd")

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgList[idx]))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(self.maskList[idx]))
        #todo normalization, shaping, proper indices

    def __len__(self):
        return len(self.imgList)
