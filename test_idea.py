from matplotlib import pyplot as plt

from dataset import Vessel12DatasetRepresentativewNoise

dspath = "./dataset"

ds = Vessel12DatasetRepresentativewNoise(dspath)
ds.auto_noise_factor = 1
ds.loadimage(range(11, 12), 300)

(img, gt) = ds.__getitem__(150)
img = img[9].cpu().detach().numpy()

fig = plt.figure(figsize=(15, 15))
plt.imshow(img)
plt.axis('off')
plt.savefig("test.png")
