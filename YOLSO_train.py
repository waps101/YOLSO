import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import numpy as np
import utils
import argparse
from data import dataset
from model import models
from utils import utils
import os

parser = argparse.ArgumentParser(description="Train a YOLSO model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--gridsize", default=16, help="number of grid cells along each side of grid for training crop")
parser.add_argument("-c", "--cellres", default=32, help="number of pixels along each side of a grid cell")
parser.add_argument("-t", "--trainprop", default=0.9, help="proportion of dataset to use for training")
parser.add_argument("-b", "--batchsize", default=16, help="batch size")
parser.add_argument("-l", "--lr", default=0.001, help="learning rate")
parser.add_argument("-r", "--resume", default=0, help="checkpoint number")
parser.add_argument("-e", "--numepochs", default=1000, help="number of epochs")
parser.add_argument("-R", "--doresize", action='store_true', help="whether to resize all training images to specified size")
parser.add_argument("-s", "--imsize", default=4724, help="whether to resize all training images to specified size")
parser.add_argument("-f", "--basefolder", default='datasets/OS_trees/', help="train data folder")

args = vars(parser.parse_args())
resume = int(args["resume"])
batch_size = int(args["batchsize"])
gridsize = int(args["gridsize"])
cellsize = int(args["cellres"])
lr = float(args["lr"])
num_epochs = int(args["numepochs"])
doresize = args["doresize"]
imsize = int(args["imsize"])
basefolder = args["basefolder"]

def is_power_of_two(n):
    return n != 0 and (n & (n - 1)) == 0

if not is_power_of_two(cellsize):
    raise ValueError("cellres must be a power of two")

if doresize:
    print('Images will be resized to {}'.format(imsize))

transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5,
                                std=0.5)
                            ])

print('Checking for GPU...')
device = utils.SelectDevice()

print('Initialising dataset...')
data = dataset.TilesData(basefolder,gridsize,transform,cellsize,doresize,imsize)
total_size = len(data)
train_size = int(float(args["trainprop"])*total_size)
test_size = total_size-train_size

print('Initialising model...')
model = models.YOLSO(cellsize,data.npointclasses,data.nregionclasses)
pad = model.getpad()
print('Padding = {:d}'.format(pad))
model = model.to(device)
data.setpad(pad*2)
os.makedirs('checkpoints',exist_ok=True)
if resume>0:
    print('Resuming from epoch {:d}'.format(resume))
    model.load_state_dict(torch.load('checkpoints/YOLSO_{:d}.pkl'.format(resume),map_location=torch.device(device)))
model.train()

data_train, data_val = random_split(data, [train_size, test_size])

print('Training on {} images, validating on {} images'.format(len(data_train),len(data_val)))

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=1)

print(model)

class_loss_func = nn.CrossEntropyLoss(reduction='none')
binary_loss_func = nn.BCELoss(reduction='none')

# Set up the optimiser
optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

# Reset scheduler to same point as resume point
if resume>0:
    for epoch in range(resume):
        if epoch % 500 == 0:
            scheduler.step()

print('Commencing training...')
iterations_per_epoch = len(train_loader)
training_accuracies = []
for epoch in range(resume,num_epochs):
  correct = 0
  total = 0
  total_loss = 0
  TPs = torch.zeros(iterations_per_epoch)
  FPs = torch.zeros(iterations_per_epoch)
  FNs = torch.zeros(iterations_per_epoch)
  regionFP = torch.zeros(iterations_per_epoch)
  regionFN = torch.zeros(iterations_per_epoch)
  regionTP = torch.zeros(iterations_per_epoch)
  regionTN = torch.zeros(iterations_per_epoch)
  pointFP = torch.zeros(iterations_per_epoch)
  pointFN = torch.zeros(iterations_per_epoch)
  pointTP = torch.zeros(iterations_per_epoch)
  pointTN = torch.zeros(iterations_per_epoch)
  wrongclass = torch.zeros(iterations_per_epoch)
  coord_errs = torch.zeros(iterations_per_epoch)
  # One epoch on the training set
  for i, (images, ispoint, classes, coords) in enumerate(train_loader):
    images, ispoint, classes, coords = images.to(device), ispoint.to(device), classes.to(device), coords.to(device)
    # classes is gridsize * gridsize and contains 0 for background,
    # 1..nregionclasses for non-point classes and
    # nregionclasses+1..nregionclasses+npointclasses for point classes

    coords = coords.float()
    xy,probs = model(images)
    # probs is gridsize * gridsize * nregionclasses+npointclasses+1
    class_loss = class_loss_func(probs,classes)

    softmaxprobs = nn.functional.softmax(probs,dim=1)
    regionconf = torch.sum(softmaxprobs[:,1:data.nregionclasses+1,:,:],dim=1) # B x H x W
    pointconf = torch.sum(softmaxprobs[:,data.nregionclasses+1:,:,:],dim=1) # B x H x W
    isregion = (classes>0) & (~ispoint)

    regions_loss = binary_loss_func(torch.clamp(regionconf,min=0,max=1),isregion.float())
    points_loss = binary_loss_func(torch.clamp(pointconf,min=0,max=1),ispoint.float())

    regions_as_points_loss = torch.mean(binary_loss_func(pointconf[isregion],torch.zeros_like(pointconf[isregion])))

    # Hard negative mining
    with torch.no_grad():
        # Include both 1. hard negatives and 2. TPs + FNs
        # negative means classes==0
        # hard means background probability isn't the maximum class
        mask = ((probs[:,0,:,:]<torch.amax(probs[:,1:,:,:],dim=1)) & (classes==0)) | (classes>0)
    #if mask.sum()>0:
    #    class_loss = class_loss[mask].mean()
    #else:
    #    class_loss = 0

    if mask.sum()>0:
        class_loss[mask] = class_loss[mask]*10.0
    class_loss = class_loss/10.0
    # class_loss[ispoint]=class_loss[ispoint]*10.0
    class_loss = class_loss.mean()
    regions_loss = regions_loss.mean()

    points_loss = points_loss.mean()

    # mask coords loss to true point cells
    with torch.no_grad():
        mask2 = ispoint.unsqueeze(1).repeat(1,2,1,1)

    coords_loss = torch.abs(coords[mask2]-xy[mask2])

    #class_loss = class_loss.mean()
    if mask2.sum()>0:
        coords_loss = coords_loss.mean()
    else:
        coords_loss = 0

    loss = class_loss + coords_loss + 10.0*points_loss + regions_loss #+ 10.0*regions_as_points_loss
    coord_errs[i]=coords_loss
    optim.zero_grad()
    if loss>0:
        loss.backward()
        optim.step()
    _, predicted = torch.max(probs, 1)
    correct += (predicted==classes).sum()

    regionFP[i] = ((~isregion) & (regionconf>=0.5)).float().sum()
    regionFN[i] = ((isregion) & (regionconf<0.5)).float().sum()
    regionTP[i] = ((isregion) & (regionconf>=0.5)).float().sum()
    regionTN[i] = ((~isregion) & (regionconf<0.5)).float().sum()

    pointFP[i] = ((~ispoint) & (pointconf>=0.5)).float().sum()
    pointFN[i] = ((ispoint) & (pointconf<0.5)).float().sum()
    pointTP[i] = ((ispoint) & (pointconf>=0.5)).float().sum()
    pointTN[i] = ((~ispoint) & (pointconf<0.5)).float().sum()

    FNs[i] = ((predicted==0) & (classes>0)).float().sum()
    TPs[i] = ((predicted==classes) & (classes>0)).float().sum()
    FPs[i] = ((predicted>0) & (classes==0)).float().sum()
    wrongclass[i] = ((predicted>0) & (predicted!=classes) & (classes>0)).float().sum()
    total += float(classes.shape[0]*classes.shape[1]*classes.shape[2])
    total_loss += loss*images.shape[0]
    #if (i+1) % 10 == 0:
       #print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}, TPs: {}, FNs: {}, FPs: {}'.format(epoch + 1, num_epochs, i + 1, iterations_per_epoch, loss.item(), TPs[i].item(), FNs[i].item(), FPs[i].item()))
       #print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}, Correct: {:.2f}'.format(epoch + 1, num_epochs, i + 1, iterations_per_epoch, loss.item()))
  total_loss /= len(data_train)
  #training_accuracies.append(correct/total)
  #print('Epoch {}: Loss={:.4f}, Bbox error (pix)={:.2f}, Train accuracy={:.4f}, TPs={}, FNs={}, FPs={}'.format(epoch+1,total_loss,coord_errs.mean().item()*cellsize,training_accuracies[-1],TPs.sum().item(),FNs.sum().item(),FPs.sum().item()))

  #print('Epoch {}: Loss={:.4f}, Bbox error (pix)={:.2f}, Correct: {:.2f}%, TPs={}, FNs={}, FPs={}, wrongP={}'.format(epoch+1,total_loss, coord_errs.mean().item()*cellsize, correct/total*100,TPs.sum().item(),FNs.sum().item(),FPs.sum().item(),wrongclass.sum().item()))

  pre_re = regionTP.sum().item() / max(0.000001,(regionTP.sum().item()+regionFP.sum().item()))
  rec_re = regionTP.sum().item() / max(0.000001,(regionTP.sum().item()+regionFN.sum().item()))
  pre_pt = pointTP.sum().item() / max(0.000001,(pointTP.sum().item()+pointFP.sum().item()))
  rec_pt = pointTP.sum().item() / max(0.000001,(pointTP.sum().item()+pointFN.sum().item()))
  print('Epoch {}: Loss={:.4f}, Bbox error (pix)={:.2f}, Pre_reg={:.4f}, Rec_reg={:.4f}, Pre_pnt={:.4f}, Rec_pnt={:.4f}, wrongP={}'.format(epoch+1,total_loss, coord_errs.mean().item()*cellsize,pre_re,rec_re,pre_pt,rec_pt,wrongclass.sum().item()))

  total = torch.sum(classes>=0)
  print('Background: {:.4f}, regions: {:.4f}, bboxes: {:.4f}'.format(
                                                                 torch.sum(classes==0)/total,
                                                                 torch.sum((classes>0)&(classes<=data.nregionclasses))/total,
                                                                 torch.sum(classes>data.nregionclasses)/total)
                                                              )

  if epoch % 500 == 0:
      torch.save(model.state_dict(), 'checkpoints/YOLSO_{:d}.pkl'.format(epoch))
      scheduler.step()

  # One epoch on the test set
  if len(data_val)>0:
      correct = 0
      total = 0
      # Switch to evaluation mode
      model.eval()
      with torch.no_grad():
        for images, ispoint, classes, coords in val_loader:
          images, ispoint, classes, coords = images.to(device), ispoint.to(device), classes.to(device), coords.to(device)
          coords = coords.float()
          xy,probs = model(images)
          _, predicted = torch.max(probs, 1)
          correct += (predicted==classes).sum()
          total += float(classes.shape[0]*classes.shape[1]*classes.shape[2])
      # Switch back to training mode
      model.train()
      print('Test accuracy at epoch {}: {:.4f}'.format(epoch+1,correct/total))

torch.save(model.state_dict(), 'checkpoints/YOLSO_final.pkl')
