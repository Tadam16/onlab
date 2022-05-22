from torch.utils.data import DataLoader

import model_dropout
import model_switch_norm
import model_switchnorm_dropout
from dataset import Vessel12DatasetRepresentative, Vessel12Dataset, Vessel12DatasetRepresentativewNoise
import model_no_normalization
import customLoss
import numpy as np
import torch.utils.data
import os
import torch
import matplotlib.pyplot as plt
import helper_functions

dspath = "./dataset"


def train(modelClass, modelname, inname = None):
    with open('stop.txt', 'w'):
        pass

    device = torch.device("cuda")

    if inname is not None:
        model = torch.load("trained_models/" + inname + ".pt", map_location=device)
    else:
        model = modelClass()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)


    lossfunc = customLoss.FalsePositiveManglerLoss(device)

    batch_size = 5
    train_iters = 100
    test_iters = 100
    epochs = 10

    seeninit = 0
    k = seeninit/batch_size
    evallosses = []
    trainlosses = []
    trainlossum = 0
    model.train()
    dataset = Vessel12DatasetRepresentativewNoise(dspath)
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
            x = x[:,0:9, :,:]
            pred = model(x)
            loss = lossfunc(pred, y)
            trainlossum += loss
            if (k % (train_iters // batch_size) == 0 and k != seeninit/batch_size):
                evallosses.append(evalmodel(testDataset, model, test_iters // batch_size, lossfunc, batch_size,
                                            device).cpu().detach().numpy())
                trainlosses.append(trainlossum.cpu().detach().numpy() / (train_iters // batch_size))
                minibatches_seen.append(k)
                # visualizelosses(evallosses, trainlosses, minibatches_seen, batch_size)
                trainlossum = 0
                print("Slices seen: " + str(k * batch_size) + " Train loss: " + str(trainlosses[-1]) + " Test loss: " + str(evallosses[-1]))

                if (not os.path.exists('stop.txt')):
                    break

                if torch.cuda.max_memory_reserved(device) > 9e9:
                    print("Memory limit exceeded, quitting...")
                    exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k += 1

    visualizelosses(evallosses, trainlosses, minibatches_seen, batch_size, modelname)
    torch.save(model, "trained_models/" + modelname + ".pt")


def evalmodel(dataset, model, limit, lossfunc, batch_size, device):
    loader = DataLoader(dataset, batch_size, True)
    lossum = 0
    k = 0
    with torch.no_grad():
        for (j, (x, y)) in enumerate(loader):
            (x, y) = (x.to(device), y.to(device))
            x = x[:, 0:9, :, :]
            pred = model(x)
            lossum += lossfunc(pred, y)
            k += 1
            if (k == limit):
                break
    return lossum / k


def visualizelosses(evalloss, trainloss, minibatches_seen, batch_size, modelname):
    plt.style.use("ggplot")
    plt.figure()
    minibatches_seen = np.multiply(minibatches_seen, batch_size)
    plt.plot(minibatches_seen, evalloss, color="red", label="test loss")
    plt.plot(minibatches_seen, trainloss, color="blue", label="train loss")
    plt.xlabel("Slices seen")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses/" + modelname + "_loss.png")
    plt.close()
    # plt.show()

if __name__ == '__main__':

    for i in range(16, 20):
        helper_functions.translate_full(dspath, i)

    #train(model_no_normalization.NestedUnet, "onlab_model_no_norm")
    #helper_functions.evaluate(dspath, "onlab_model_no_norm")

    #train(model_switch_norm.NestedUnet, "onlab_model_switchnorm")
    #helper_functions.evaluate(dspath, "onlab_model_switchnorm")

    #train(model_dropout.NestedUnet, "onlab_model_dropout")
    #helper_functions.evaluate(dspath, "onlab_model_dropout")

    #train(model_switchnorm_dropout.NestedUnet, "model_switchnorm_dropout_autoimg_cut_fp_punish", "model_switchnorm_dropout_autoimg_cut_trained")
    #helper_functions.evaluate(dspath, "model_switchnorm_dropout_autoimg_cut_fp_punish")

    print("Training completed!")
    print(torch.cuda.max_memory_reserved(torch.device("cuda")))
