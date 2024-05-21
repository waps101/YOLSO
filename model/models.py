import torch
import torch.nn as nn

class YOLSO(nn.Module):
  def __init__(self,cellsize,npointclasses,nregionclasses):
    super(YOLSO, self).__init__()

    self.relu = nn.ReLU()

    if cellsize==8:
        channels = [64,128,196,256]

    if cellsize==16:
        channels = [64,96,128,196,256]

    if cellsize==32:
        channels = [32,64,96,128,196,256]

    if cellsize==64:
        channels = [32,64,96,128,196,256,512]

    if cellsize==128:
        channels = [32,64,96,128,196,256,384,512]

    layers=[
      nn.Conv2d(in_channels = 1, out_channels = channels[0], kernel_size = 7, stride = 1, padding = 0, bias=False),
      nn.BatchNorm2d(channels[0]),
      nn.ReLU(),
      nn.Conv2d(in_channels = channels[0], out_channels = channels[0], kernel_size = 3, stride = 1, padding = 0, bias=False),
      nn.BatchNorm2d(channels[0]),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2)
      ]

    for i in range(1,len(channels)-1):
        layers.extend(
          [
          nn.Conv2d(in_channels = channels[i-1], out_channels = channels[i], kernel_size = 3, stride = 1, padding = 0, bias=False),
          nn.BatchNorm2d(channels[i]),
          nn.ReLU(),
          nn.Conv2d(in_channels = channels[i], out_channels = channels[i], kernel_size = 3, stride = 1, padding = 0, bias=False),
          nn.BatchNorm2d(channels[i]),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2, stride = 2)
          ]
          )

    # MLP implemented as 1x1 convolutions
    layers.extend(
        [
        nn.Conv2d(in_channels = channels[-2], out_channels = channels[-1], kernel_size = 3, stride = 1, padding = 0, bias=False), # try 3x3
        nn.BatchNorm2d(channels[-1]),
        nn.ReLU(),
        nn.Conv2d(in_channels = channels[-1], out_channels = 3+npointclasses+nregionclasses, kernel_size = 1, stride = 1, padding = 0)
        ]
        )

    self.FullyConvNet = nn.Sequential(*layers)


    self.sigmoid = nn.Sigmoid()

    # Compute padding from max pool and conv layers
    d = 0
    P = 0
    for layer in self.FullyConvNet.children():
      if isinstance(layer, nn.Conv2d):
        k = layer.kernel_size[0]
        P += (k-1)/2*2**d
      if isinstance(layer, nn.MaxPool2d):
        d += 1
    self.pad = int(P)

  def forward(self, x):
    x = self.FullyConvNet(x) # Output size: B x 3 x Hgrid x Wgrid
    # Channel 0-1: coordinates of bounding box centre (within grid cell, normalised 0..1)
    # Channel 2-end: raw logits for classes, including additional class for background
    xy = self.sigmoid(x[:,0:2,:,:]) # Size: B x 2 x Hgrid x Wgrid
    probs = x[:,2:,:,:] # Size: B x npointclasses+nregionclasses+1 x Hgrid x Wgrid
    return xy,probs

  def getpad(self):
      return self.pad
