import matplotlib.pyplot as plt

device = torch.device("cpu")

model = torch.load("drive/MyDrive/onlab_model_low_mem_ce_dropout_bs5.pt", map_location=device)
ds = Vessel12Dataset(dspath)
ds.loadimage(16)
evalindex = 200

(x, img, auto, mask) = ds.evalitem(evalindex)

with torch.no_grad():
    model.eval()
    x = x.to(device)
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

model = torch.load("drive/MyDrive/onlab_model_test.pt")
ds = Vessel12Dataset(dspath)
ds.loadimage(2);
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

scipy.io.savemat(r'drive/Shareddrives/Enforced Privacy Demotion/dam_onlab/recognition.mat', {"img": result})

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