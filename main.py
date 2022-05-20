from torch.utils.data import DataLoader
from dataset import Vessel12DatasetRepresentative, Vessel12Dataset
from model import NestedUnet
import numpy as np
import torch.utils.data
import os
import torch
import matplotlib.pyplot as plt

dspath = "./dataset"

def train():
    with open('stop.txt', 'w'):
        pass

    device = torch.device("cuda")
    model = NestedUnet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lossfunc = torch.nn.BCELoss()

    batch_size = 5
    train_iters = 100
    test_iters = 100
    epochs = 10

    k = 0
    evallosses = []
    trainlosses = []
    trainlossum = 0
    model.train()
    dataset = Vessel12DatasetRepresentative(dspath)
    testDataset = Vessel12DatasetRepresentative(dspath)
    testDataset.loadimage(list(range(11, 16)), 100)
    minibatches_seen = []

    # training
    for i in range(epochs):
        if (not os.path.exists('stop.txt')):
            break

        dataset.loadimage(list(range(0, 11)), 100)
        loader = DataLoader(dataset, batch_size, True)

        for (j, (x, y)) in enumerate(loader):

            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossfunc(pred, y)
            trainlossum += loss
            if (k % (train_iters // batch_size) == 0 and k != 0):
                evallosses.append(evalmodel(testDataset, model, test_iters // batch_size, lossfunc, batch_size,
                                            device).cpu().detach().numpy())
                trainlosses.append(trainlossum.cpu().detach().numpy() / (train_iters // batch_size))
                minibatches_seen.append(k)
                visualizelosses(evallosses, trainlosses, minibatches_seen, batch_size)
                trainlossum = 0
                print("Train loss: " + str(trainlosses[-1]) + " Test loss: " + str(evallosses[-1]))
                if (not os.path.exists('stop.txt')):
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k += 1

    visualizelosses(evallosses, trainlosses, minibatches_seen, batch_size)
    torch.save(model, "trained_models/onlab_model_low_mem_ce_dropout_bs5.pt")


def evalmodel(dataset, model, limit, lossfunc, batch_size, device):
    loader = DataLoader(dataset, batch_size, True)
    lossum = 0
    k = 0
    with torch.no_grad():
        for (j, (x, y)) in enumerate(loader):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            lossum += lossfunc(pred, y)
            k += 1
            if (k == limit):
                break

    return lossum / k


def visualizelosses(evalloss, trainloss, minibatches_seen, batch_size):
    plt.style.use("ggplot")
    plt.figure()
    minibatches_seen = np.multiply(minibatches_seen, batch_size)
    plt.plot(minibatches_seen, evalloss, color="red", label="test loss")
    plt.plot(minibatches_seen, trainloss, color="blue", label="train loss")
    plt.xlabel("Slices seen")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
    #plt.show()

if __name__ == '__main__':
    train()
    print("Training completed!")
    print(torch.cuda.max_memory_reserved(torch.device("cuda")))