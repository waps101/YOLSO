import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import utils
from data import dataset
from model import models
from utils import utils
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process a dataset using a trained YOLSO model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--gridsize", default=16, help="number of grid cells along each side of grid for training crop")
parser.add_argument("-c", "--cellres", default=32, help="number of pixels along each side of a grid cell - must be power of 2")
parser.add_argument("-R", "--doresize", action='store_true', help="whether to resize all training images to specified size")
parser.add_argument("-s", "--imsize", default=4724, help="whether to resize all training images to specified size")
parser.add_argument("-p", "--npointclasses", default=3, help="number of point classes")
parser.add_argument("-r", "--nregionclasses", default=5, help="number of region classes")
parser.add_argument("-b", "--basefolder", default='datasets/test/', help="test data path")
parser.add_argument("-i", "--imagesfolder", default='images/', help="test images folder")
parser.add_argument("-o", "--outputfolder", default='results/', help="test output folder")
parser.add_argument("-w", "--weightspath", default='models/YOLSO_OS.pkl', help="path to trained model weights")
parser.add_argument("-B", "--bboxwidth", default=48, help="bounding box width")

args = vars(parser.parse_args())
cellsize = int(args["cellres"])
path = args["basefolder"]
npointclasses = int(args["npointclasses"])
nregionclasses = int(args["nregionclasses"])
weightspath = args["weightspath"]
imagesfolder = args["imagesfolder"]
outputfolder = args["outputfolder"]
doresize = args["doresize"]
imsize = int(args["imsize"])
bboxwidth = int(args["bboxwidth"])

transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5,
                                std=0.5)
                            ])

device = utils.SelectDevice()

print('Initialising test dataset...')
data_test = dataset.TilesDataTest(path+imagesfolder,transform,cellsize,doresize,imsize)

print('Initialising model...')
model = models.YOLSO(cellsize,npointclasses,nregionclasses)
pad = model.getpad()*2
print('Padding = {:d}'.format(pad))
model = model.to(device)
data_test.setpad(pad)
model.load_state_dict(torch.load(weightspath,map_location=torch.device(device)))
model.eval()

os.makedirs(path+outputfolder,exist_ok=True)

with torch.no_grad():
    for i in range(len(data_test)):
        # h is the original size of the image while image may have been resized if the doresize option is set to True
        image,filename,h,gridsize = data_test[i]

        t1 = time.perf_counter()
        image = image.to(device)
        image = image.unsqueeze(0) # 1 x 1 x H x W

        xy,logits = model(image)
        t2 = time.perf_counter()
        print('{:d}: {:.4f}s'.format(i,t2-t1))

        probs = nn.functional.softmax(logits,dim=1) # B x nclasses x H x W
        regionconf = torch.sum(probs[:,1:nregionclasses+1,:,:],dim=1) # B x H x W
        pointconf = torch.sum(probs[:,nregionclasses+1:,:,:],dim=1) # B x H x W

        confs, predicted = torch.max(probs, 1) # B x H x W

        confs, predicted = torch.max(logits, 1) # B x H x W

        for x in range(predicted.shape[1]):
            for y in range(predicted.shape[2]):
                if pointconf[0,x,y]>=0.5:
                    # This is a point
                    confs[0,x,y] = pointconf[0,x,y] # 0 because batch size of 1
                    logits[0,0:nregionclasses+1,x,y]=torch.min(logits)
                elif regionconf[0,x,y]>=0.5:
                    # This is a region
                    confs[0,x,y] = regionconf[0,x,y]
                    logits[0,0,x,y]=torch.min(logits)
                    logits[0,nregionclasses+1:,x,y]=torch.min(logits)
        _, predicted = torch.max(logits, 1) # B x H x W

        P = utils.GetBoundingBoxes(predicted.squeeze(),confs.squeeze(),xy.squeeze(),gridsize,cellsize,nregionclasses,bboxwidth)
        print('{:d} bounding boxes detected'.format(P.shape[0]))
        if P.shape[0]>0:
            P = utils.NMS(P,0.5)
        print('{:d} bounding boxes after NMS'.format(len(P)))
        utils.DrawOutput(image.squeeze(),predicted.squeeze(),xy.squeeze(),gridsize,cellsize,pad,path+outputfolder+filename[:-4]+'.tif',nregionclasses,h,doresize,imsize,bboxwidth,crop=True)
        utils.SaveRaster(predicted.squeeze(),xy.squeeze(),gridsize,cellsize,pad,path+outputfolder+filename[:-4]+'_raster.tif',nregionclasses,h,doresize,imsize)
        utils.SaveBoundingBoxes(P,path+outputfolder+filename[:-4]+'.txt',nregionclasses,h,doresize,imsize,bboxwidth)
