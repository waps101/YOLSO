from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import csv
import numpy as np
import torch
from matplotlib import path
import math

class TilesData(Dataset):
  """Custom OS map tree tiles dataset."""

  def __init__(self, basefolder, gridsize, transform, cellsize, doresize, imsize):

    # Build dictionary relating region and point class names to ID (starting from 1 as background is 0)
    self.regionclasses = dict()
    count = 1
    with open(basefolder+'/regionclasses.txt') as f:
        line = f.readline()
        while line:
            self.regionclasses[line.rstrip()]=count
            line = f.readline()
            count += 1
    self.nregionclasses = len(self.regionclasses)
    print(self.nregionclasses,'region classes')
    self.pointclasses = dict()
    with open(basefolder+'/pointclasses.txt') as f:
        line = f.readline()
        while line:
            self.pointclasses[line.rstrip()]=count
            line = f.readline()
            count += 1 # We did not reset count so continues from nregionclasses
    self.npointclasses = len(self.pointclasses)
    print(self.npointclasses,'point classes')

    print(self.regionclasses)
    print(self.pointclasses)

    # Load the bounding box centre annotations
    all_files = os.listdir(basefolder+'/points')
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    self.allpoints = []
    self.allregions = []
    self.transform = transform
    self.images = []
    self.cellsize = cellsize
    self.gridsize = gridsize
    total = 0
    total_region = 0
    for i in range(len(csv_files)):

      # Load image
      img = Image.open(basefolder+'/tiles/'+csv_files[i][:-4]+'.tif')

      # Load point coordinates
      pts = np.array([])
      with open(basefolder+'/points/'+csv_files[i], newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            pointclass = self.pointclasses[row[2]]
            new_row = [pointclass, x, y]
            if pts.size == 0:
                pts = np.array([new_row])
            else:
                pts = np.vstack((pts, new_row))

      # Check if we need to resize
      if doresize and (img.size[0] != imsize):
          scale = imsize / float(img.size[0])
          hsize = int((float(img.size[1]) * float(scale)))
          img = img.resize((imsize, hsize))

          # We are resizing the image so need to scale point coordinates accordingly
          if pts.size > 0:
              pts[:,1:3]=pts[:,1:3] * scale

      total = total+pts.shape[0]
      self.allpoints.append(pts)

      img = self.transform(img)
      self.images.append(img)

      # Load region coordinates
      regions = []
      with open(basefolder+'/regions/'+csv_files[i][:-4]+'.txt') as f:
          line = f.readline()
          while line:
              region = eval(line)
              regions.append(region)
              line = f.readline()
          total_region += len(regions)
      self.allregions.append(regions)

    print('Total point samples = {}'.format(total))
    print('Total regions = {}'.format(total_region))

  def __len__(self):
    return len(self.allpoints)

  def setpad(self,pad):
      self.pad = pad
      self.dim = self.gridsize*self.cellsize+pad
      print('Dimensions',self.dim)

  def nregionclasses(self):
      return self.nregionclasses

  def npointclasses(self):
      return self.npointclasses

  def __getitem__(self, idx):

    bbox = self.allpoints[idx] # npts x 3, with 0: class, 1..2: coordinate
    regions = self.allregions[idx] # list containing lists, each of which contains class name followed by list of coordinates as 2D tuples
    img = self.images[idx]

    # Random crop of size self.dim^2
    top = np.random.randint(img.shape[1]-self.dim)
    left = np.random.randint(img.shape[2]-self.dim)
    img = img[:,top:top+self.dim,left:left+self.dim]

    classes = torch.zeros((int(self.gridsize),int(self.gridsize)),dtype=torch.long)
    coords = torch.zeros((2,int(self.gridsize),int(self.gridsize)))

    # Create a grid
    xv,yv = np.meshgrid(np.linspace(0.5,self.gridsize-0.5,self.gridsize),np.linspace(0.5,self.gridsize-0.5,self.gridsize))
    if len(regions)>0:
        # For each region polygon, find which grid point centres lie inside the polygon
        for i in range(len(regions)):
            # Convert list of tuples to n x 2 matrix and apply scaling to pixel space
            pts = np.asarray(regions[i][1:])*4724.0 #*float(img.shape[1])
            # Adjust region boundary coordinates to crop and scale to grid
            pts[:,0] = (pts[:,0]-left-self.pad/2)/self.cellsize
            pts[:,1] = (pts[:,1]-top-self.pad/2)/self.cellsize
            # Pts now in units of grid
            # Convert back to list of tuples
            pts = list(map(tuple,pts))
            p = path.Path(pts)
            flags = p.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))
            # Unflatten
            flags = torch.from_numpy(flags.reshape(self.gridsize,self.gridsize))
            # Put class number into appropriate cells
            classes[flags] = self.regionclasses[regions[i][0]]

    # Adjust bounding box coordinates to crop
    if bbox.shape[0]>0:
      bbox[:,1] = bbox[:,1]-left-self.pad/2
      bbox[:,2] = bbox[:,2]-top-self.pad/2
      mask = (bbox[:,1]>=0) & (bbox[:,1]<self.gridsize*self.cellsize) & (bbox[:,2]>=0) & (bbox[:,2]<self.gridsize*self.cellsize)
      bbox = torch.tensor(bbox[mask,:])

      # Round bbox centres to cell coordinates
      bbox_cells = torch.floor(bbox[:,1:3]/self.cellsize).to(torch.int32)
      #print(bbox_cells)
      if bbox.shape[0]>0:
        for i in range(bbox.shape[0]):
          coords[:,bbox_cells[i,1],bbox_cells[i,0]]=(bbox[i,1:3]-bbox_cells[i,:]*self.cellsize)/(self.cellsize-1)
          classes[bbox_cells[i,1],bbox_cells[i,0]] = bbox[i,0]

    ispoint = classes > self.nregionclasses
    #bbox = torch.tensor(bbox)
    return img, ispoint, classes, coords

class TilesDataTest(Dataset):
    """Data loader for test data. Only returns image and filename. Loads image on demand since dataset could be very large."""

    def __init__(self, imagesfolder, transform, cellsize, doresize, imsize):

        all_files = os.listdir(imagesfolder)
        image_files = list(filter(lambda f: f.endswith('.tif'), all_files))
        self.fnames = []
        self.imagesfolder = imagesfolder
        self.transform = transform
        self.cellsize = cellsize
        self.doresize = doresize
        self.imsize = imsize
        self.image_files = image_files

        for i in range(len(image_files)):
            self.fnames.append(imagesfolder+image_files[i])

    def __len__(self):
        return len(self.fnames)

    def setpad(self,pad):
        self.pad = pad

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx]) # 2D if grayscale, 3D if colour
        h = img.size[-2]

        # Check if we need to resize
        if self.doresize and (img.size[0] != self.imsize):
            scale = self.imsize / float(img.size[0])
            hsize = int((float(img.size[1]) * float(scale)))
            img = img.resize((self.imsize, hsize))

        R = self.cellsize
        W = img.size[0]
        Tw = R*(1-W/R+math.floor(W/R))

        gridsize = int((W+Tw)/R)

        dim = gridsize*self.cellsize+self.pad

        img = self.transform(img) # Now either 1 x H x W or 3 x H x W
        image = torch.ones(img.shape[0],dim,dim)
        image[:,int(self.pad/2):int(self.pad/2)+img.shape[-2],int(self.pad/2):int(self.pad/2)+img.shape[-1]]=img

        return image,self.image_files[idx],h,gridsize
