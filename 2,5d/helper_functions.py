import matplotlib.pyplot as plt
import numpy as np
import scipy
import  torch
from torch.utils.data import DataLoader

from dataset import Vessel12DatasetRepresentative
from dataset import Vessel12Dataset

device = torch.device("cuda")

def evaluate(dspath, modelname):
    model = torch.load("trained_models/" + modelname + ".pt", map_location=device)
    ds = Vessel12Dataset(dspath)
    ds.loadimage(16)
    evalindex = 200

    (x, img, auto, mask) = ds.evalitem(evalindex)

    with torch.no_grad():
        model.eval()
        x = x.to(device)
        x = x[:,0:9, :,:]
        out = model(x)

        out = out.cpu().detach().numpy()

    fig = plt.figure(figsize=(15, 15))
    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('CT image')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(auto)
    plt.axis('off')
    plt.title('Auto generated image')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(out[0, 0, :, :])
    plt.axis('off')
    plt.title('Predicted image')

    fig.add_subplot(rows, columns, 4)
    plt.imshow(mask[0, 0, :, :])
    plt.axis('off')
    plt.title('Ground truth')

    plt.figure(fig.number)
    plt.savefig("evals/" + modelname + "_evaluate.png")
    plt.close()

def translate_full(dspath, idx):
    model = torch.load("trained_models/model_switchnorm_dropout_autoimg_cut_fp_punish.pt")
    ds = Vessel12Dataset(dspath)
    ds.loadimage(idx)
    dl = DataLoader(ds, shuffle=False)
    result = np.empty((ds.shape()[0], 512, 512))

    with torch.no_grad():
        for (j, (x, y)) in enumerate(dl):
            device = torch.device("cuda")
            x = x.to(device)
            pred = model(x)
            pred = pred.cpu().detach().numpy()
            pred = pred[0, 0, :, :]
            print(j)
            result[j, :, :] = pred

    scipy.io.savemat(r'fld/processed_cts/ct' + str(idx), {"img": result}, do_compression=True)

def dataset_distribution(dspath):
    ds = Vessel12DatasetRepresentative(dspath)
    ds.loadimage(list(range(11, 16)), 100)

    fig = plt.figure(figsize = (15, 15))
    rows = 1
    columns = 2

    data = ds.getimg()

    fig.add_subplot(rows, columns, 1)
    plt.hist(data.flatten(), bins=256, histtype = 'step', density = True)
    plt.title('Testing data histogram')

    ds.loadimage(list(range(0, 11)), 100)

    data = ds.getimg()

    fig.add_subplot(rows, columns, 2)
    plt.hist(data.flatten(), bins=256, histtype = 'step', density = True)
    plt.title('Training data histogram')

def loss_on_autoimg(dspath):
    #Average loss for helper slices as if they were the output
    ds = Vessel12DatasetRepresentative(dspath)
    ds.loadimage(list(range(11, 16)), 100)
    loader = DataLoader(ds, 5, True)
    lossum = 0
    k = 0
    device = torch.device("cuda")
    lossfunc = torch.nn.BCELoss()

    with torch.no_grad():
      for (j, (x, y)) in enumerate(loader):
                (x, y) = (x.to(device), y.to(device))
                pred = x[:, 9, :, :].reshape([5,1,512,512])
                lossum += lossfunc(pred, y)
                k += 1

    lossum = lossum / k

    print(lossum.item())