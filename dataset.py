import numpy as np
import torch.utils.data
import torch
import SimpleITK as sitk
import glob

import scipy.io

class Vessel12Dataset(torch.utils.data.Dataset):

    def __init__(self, dspath = "."):
        self.mask = None
        self.img = None
        self.autoimg = None
        self.imgList = glob.glob(dspath + "/imgs/*.mhd")
        self.autoimgList = glob.glob(dspath + "/autoimgs/*.mat")
        self.maskList = glob.glob(dspath + "/masks/*.mhd")
        self.imgList.sort()
        self.autoimgList.sort()
        self.maskList.sort()

    def loadimage(self, idx):
        self.img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgList[idx]))
        self.img = (np.clip(self.img, -1200, 600) + 1200) / 1800.0
        self.autoimg = scipy.io.loadmat(self.autoimgList[idx])["maszk"]
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(self.maskList[idx]))

    def shape(self):
        return self.img.shape

    def evalitem(self, idx):
        idx = idx + 4

        resimg = self.img[idx - 4:idx + 5, :, :]
        resautoimg = self.autoimg[idx, :, :]
        resimg2 = np.concatenate((resimg, resautoimg.reshape([1,512,512])), axis=0).reshape([1, 10, 512, 512])
        resmask = self.mask[idx, :, :].reshape([1, 1, 512, 512])
        return torch.from_numpy(resimg2).float(), resimg[4, :, :], resautoimg, resmask

    def __getitem__(self, idx):
        idx = idx + 4

        resimg = self.img[idx - 4:idx + 5, :, :]
        resautoimg = self.autoimg[idx, :, :]
        resimg = np.concatenate((resimg, resautoimg.reshape([1,512,512])), axis=0)
        resmask = self.mask[idx, :, :].reshape([1, 512, 512])

        return torch.from_numpy(resimg).float(), torch.from_numpy(resmask).float()

    def __len__(self):
        if self.img is None:
            return 0
        return self.img.shape[0] - 8


import random
class Vessel12DatasetRepresentative(torch.utils.data.Dataset):

    def __init__(self, dspath = "."):
        self.mask = None
        self.img = None
        self.autoimg = None
        self.loaded = 0
        self.length = 0
        self.imgList = glob.glob(dspath + "/imgs/*.mhd")
        self.autoimgList = glob.glob(dspath + "/autoimgs/*.mat")
        self.maskList = glob.glob(dspath + "/masks/*.mhd")
        self.imgList.sort()
        self.autoimgList.sort()
        self.maskList.sort()

    def getimg(self):
      return self.img

    def loadimage(self, indices, length):
        self.img = None
        self.autoimg = None
        self.mask = None
        self.loaded = len(indices)
        self.length = length

        for idx in indices:
          img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgList[idx]))
          img = (np.clip(img, -1200, 600) + 1200) / 1800.0
          autoimg = scipy.io.loadmat(self.autoimgList[idx])["maszk"]
          mask = sitk.GetArrayFromImage(sitk.ReadImage(self.maskList[idx]))
          size = img.shape[0] - length
          startidx = random.randrange(0, size)
          if(self.img is None):
            self.img = img[startidx : startidx + length, :,:]
            self.autoimg = autoimg[startidx : startidx + length, : , :]
            self.mask = mask[startidx : startidx + length, :, :]
          else:
            self.img = np.concatenate((self.img, img[startidx : startidx + length, :,:]))
            self.autoimg = np.concatenate((self.autoimg, autoimg[startidx : startidx + length, : , :]))
            self.mask = np.concatenate((self.mask, mask[startidx : startidx + length, :, :]))

    def __getitem__(self, idx):
      idx = idx // (self.length - 8) * self.length + 4 + idx % (self.length - 8)
      resimg = self.img[idx - 4:idx + 5, :, :]
      resautoimg = self.autoimg[idx, :, :]
      resimg = np.concatenate((resimg, resautoimg.reshape([1,512,512])), axis=0)
      resmask = self.mask[idx, :, :].reshape([1, 512, 512])

      return torch.from_numpy(resimg).float(), torch.from_numpy(resmask).float()

    def __len__(self):
        if self.img is None:
            return 0
        return (self.length - 8) * self.loaded

class Vessel12DatasetRepresentativewNoise(torch.utils.data.Dataset):

    def __init__(self, dspath = "."):
        self.mask = None
        self.img = None
        self.autoimg = None
        self.loaded = 0
        self.length = 0
        self.imgList = glob.glob(dspath + "/imgs/*.mhd")
        self.autoimgList = glob.glob(dspath + "/autoimgs/*.mat")
        self.maskList = glob.glob(dspath + "/masks/*.mhd")
        self.imgList.sort()
        self.autoimgList.sort()
        self.maskList.sort()
        self.auto_noise_factor = 0

    def getimg(self):
      return self.img

    def loadimage(self, indices, length):
        self.img = None
        self.autoimg = None
        self.mask = None
        self.loaded = len(indices)
        self.length = length

        for idx in indices:
          img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgList[idx]))
          img = (np.clip(img, -1200, 600) + 1200) / 1800.0
          autoimg = scipy.io.loadmat(self.autoimgList[idx])["maszk"]
          mask = sitk.GetArrayFromImage(sitk.ReadImage(self.maskList[idx]))
          size = img.shape[0] - length
          startidx = random.randrange(0, size)
          if(self.img is None):
            self.img = img[startidx : startidx + length, :,:]
            self.autoimg = autoimg[startidx : startidx + length, : , :]
            self.mask = mask[startidx : startidx + length, :, :]
          else:
            self.img = np.concatenate((self.img, img[startidx : startidx + length, :,:]))
            self.autoimg = np.concatenate((self.autoimg, autoimg[startidx : startidx + length, : , :]))
            self.mask = np.concatenate((self.mask, mask[startidx : startidx + length, :, :]))

    def __getitem__(self, idx):
      idx = idx // (self.length - 8) * self.length + 4 + idx % (self.length - 8)
      resimg = self.img[idx - 4:idx + 5, :, :]
      resautoimg = self.autoimg[idx, :, :]
      noise = np.random.rand(512, 512)
      resautoimg = self.auto_noise_factor * noise + (1 - self.auto_noise_factor) * resautoimg
      resimg = np.concatenate((resimg, resautoimg.reshape([1,512,512])), axis=0)
      resmask = self.mask[idx, :, :].reshape([1, 512, 512])

      return torch.from_numpy(resimg).float(), torch.from_numpy(resmask).float()

    def __len__(self):
        if self.img is None:
            return 0
        return (self.length - 8) * self.loaded