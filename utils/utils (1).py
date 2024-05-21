import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from matplotlib import path

def SelectDevice():
    # Function to automatically select device (CPU or GPU with most free memory)
    # Returns a torch device
    # OS call to nvidia-smi can probably be replaced with nvidia python library
    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp_free_gpus')
        with open('tmp_free_gpus', 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                                for idx,x in enumerate(frees) ]
        idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
        idx = idx_freeMemory_pair[0][0]
        device = torch.device("cuda:" + str(idx))
        print("Using GPU idx: " + str(idx))
        os.remove('tmp_free_gpus')
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def SaveRaster(classes,coords,gridsize,cellsize,pad,fname,nregionclasses,h,doresize,imsize):
    raster_out = torch.zeros((h,h))
    m = nn.Upsample(size=(gridsize*cellsize,gridsize*cellsize),mode='bilinear')
    for i in range(nregionclasses):
        mask = (classes==i+1)*1.0
        raster = m(mask.unsqueeze(0).unsqueeze(0).float())
        if doresize:
            raster = raster[:,:,0:imsize,0:imsize]
            mm = nn.Upsample(size=(h,h),mode='bilinear')
            raster = mm(raster)
        else:
            raster = raster[:,:,0:h,0:h]
        raster_out[raster.squeeze(0).squeeze(0)>0.5]=i+1
    raster = raster_out.detach().cpu().numpy().astype(np.uint8)

    cv2.imwrite(fname,raster)

def DrawOutput(images,classes,coords,gridsize,cellsize,pad,fname,nregionclasses,h,doresize,imsize,bboxwidth,crop=False):
    colours = np.random.randint(0, 256, size=(int(torch.max(classes)), 3))

    images = images.repeat(1,3,1,1)
    images = images*0.5+0.5
    images = images.squeeze(0)
    im = (255*images.permute(1,2,0)).to(torch.uint8).detach().cpu().numpy()
    im2 = im.copy()
    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes = np.zeros_like(im2, np.uint8)
    w = float(cellsize-1.0)/2.0
    for y in range(gridsize):
        for x in range(gridsize):
          cx = (x*cellsize+cellsize/2 + pad/2)
          cy = (y*cellsize+cellsize/2 + pad/2)
          if classes[y,x]>0:
              if classes[y,x]<=nregionclasses:
                  shapes = cv2.rectangle(shapes,(int(cx-w),int(cy-w)),(int(cx+w),int(cy+w)), colours[int(classes[y,x])-1,:].tolist(), cv2.FILLED)

    out = im2.copy()
    alpha = 0.5
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(im2, alpha, shapes, 1 - alpha, 0)[mask]

    w = (float(bboxwidth)-1)/2.0

    for y in range(gridsize):
        for x in range(gridsize):
            if classes[y,x]>nregionclasses:
                cx = (coords[0,y,x]*(cellsize-1) + x*cellsize + pad/2).detach().cpu().numpy()
                cy = (coords[1,y,x]*(cellsize-1) + y*cellsize + pad/2).detach().cpu().numpy()
                out = cv2.rectangle(out,(int(cx-w),int(cy-w)),(int(cx+w),int(cy+w)), colours[int(classes[y,x])-1,:].tolist(), 2)

    if crop:
        if doresize:
            out = out[int(pad/2):int(pad/2)+imsize,int(pad/2):int(pad/2)+imsize,:]
        else:
            out = out[int(pad/2):int(pad/2)+h,int(pad/2):int(pad/2)+h,:]

    cv2.imwrite(fname,cv2.cvtColor(out,cv2.COLOR_RGB2BGR))

def GetBoundingBoxes(classes,confs,coords,gridsize,cellsize,nregionclasses,bboxwidth):
    P = torch.tensor([])
    w = float(bboxwidth)/2.0
    for y in range(gridsize):
        for x in range(gridsize):
            if classes[y,x]>nregionclasses:
                cx = (coords[0,y,x]*(cellsize-1) + x*cellsize).detach().cpu()
                cy = (coords[1,y,x]*(cellsize-1) + y*cellsize).detach().cpu()
                bbox = torch.zeros(1,6)
                bbox[0,0] = cx-w
                bbox[0,1] = cy-w
                bbox[0,2] = cx+w
                bbox[0,3] = cy+w
                bbox[0,4] = confs[y,x].detach().cpu()
                bbox[0,5] = classes[y,x].detach().cpu()
                P = torch.cat((P,bbox), 0)
    return P

def SaveBoundingBoxes(P,fname,nregionclasses,h,doresize,imsize,bboxwidth):
    f = open(fname, 'w')
    if doresize:
        w = float(bboxwidth)/float(imsize)
        for i in range(len(P)):
            cx = 0.5*(P[i][0]+P[i][2])
            cy = 0.5*(P[i][1]+P[i][3])
            f.write('{:d} {:.10f} {:.10f} {:.10f} {:.10f}\n'.format(int(P[i][5]-nregionclasses-1),cx/float(imsize),cy/float(imsize),w,w))
    else:
        w = float(bboxwidth)/float(h)
        for i in range(len(P)):
            cx = 0.5*(P[i][0]+P[i][2])
            cy = 0.5*(P[i][1]+P[i][3])
            f.write('{:d} {:.10f} {:.10f} {:.10f} {:.10f}\n'.format(int(P[i][5]-nregionclasses-1),cx/float(h),cy/float(h),w,w))

    f.close()


"""
Apply non-maximum suppression to avoid detecting too many
overlapping bounding boxes for a given object.
Args:
    boxes: (tensor) The location preds for the image
        along with the class predscores, Shape: [num_boxes,5].
    thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
Returns:
    A list of filtered boxes, Shape: [ , 5]
"""
def NMS(P,thresh_iou=0.5):

    # we extract coordinates for every
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
    # we extract the confidence scores as well
    scores = P[:, 4]
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
    # initialise an empty list for
    # filtered prediction boxes
    keep = []
    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
        # push S in filtered predictions list
        keep.append(P[idx])
        # remove S from P
        order = order[:-1]
        # sanity check
        if len(order) == 0:
            break
        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        # find height and width of the intersection boxes
        wi = xx2 - xx1
        hi = yy2 - yy1
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        wi = torch.clamp(wi, min=0.0)
        hi = torch.clamp(hi, min=0.0)
        # find the intersection area
        inter = wi*hi
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order)
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        # find the IoU of every prediction in P with S
        IoU = inter / union
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep
