# [You Only Look for a Symbol Once: An Object Detector for Symbols and Regions in Documents](https://eprints.whiterose.ac.uk/198859/1/_ICDAR_YOLSO.pdf)

 [William A. P. Smith](https://www-users.cs.york.ac.uk/wsmith) and [Toby Pillatt](https://www.york.ac.uk/archaeology/people/toby-pillatt)
 <br/>
 University of York, UK
 <br/>
 <br/>
#### [ICDAR2023]

<br/>

## Abstract

We present YOLSO, a single stage object detector specialised for the detection of fixed size, non-uniform (e.g. hand-drawn or stamped) symbols in maps and other historical documents. Like YOLO, a single convolutional neural network predicts class probabilities and bounding boxes over a grid that exploits context surrounding an object of interest. However, our specialised approach differs from YOLO in several ways. We can assume symbols of a fixed scale and so need only predict bounding box centres, not dimensions. We can design the grid size and receptive field of a grid cell to be appropriate to the known scale of the symbols. Since maps have no meaningful boundary, we use a fully convolutional architecture applicable to any resolution and avoid introducing unwanted boundary dependency by using no padding. We extend the method to also perform coarse segmentation of regions indicated by symbols using the same single architecture. We evaluate our approach on the task of detecting symbols denoting free-standing trees and wooded regions in first edition Ordnance Survey maps and make the corresponding dataset as well as our implementation publicly available.

## PyTorch implementation

CODE CLEANING AND UPLOAD IN PROGRESS

### Training data format

In order to train a YOLSO model, you need a set of images (e.g. map tiles) annotated with bounding boxes with classes and/or polygonal regions with classes. The method predicts only bounding box centres - not width and height. So you only need centre annotations. A training dataset should be structured as follows:

    <base_folder>
    pointclasses.txt     # list of bounding box class names
    regionclasses.txt    # list of region class names
    |-- tiles
        |-- 000.tif      # map or document tiles as images
        |-- 001.tif
        ...
    |-- points
        |-- 000.txt      # bounding box centre and symbol class annotations
        |-- 001.txt
        ...
    |-- regions
        |-- 000.txt      # region polygon and class annotations
        |-- 001.txt
        ...


### Training a model

```
    usage: YOLSO_train.py [-h] [-g GRIDSIZE] [-c CELLRES] [-t TRAINPROP] [-b BATCHSIZE] [-l LR] [-r RESUME] [-e NUMEPOCHS] [-R]
                      [-s IMSIZE] [-f BASEFOLDER]

    Train a YOLSO model
    
    options:
      -h, --help            show this help message and exit
      -g GRIDSIZE, --gridsize GRIDSIZE
                            number of grid cells along each side of grid for training crop (default: 16)
      -c CELLRES, --cellres CELLRES
                            number of pixels along each side of a grid cell (default: 32)
      -t TRAINPROP, --trainprop TRAINPROP
                            proportion of dataset to use for training (default: 0.9)
      -b BATCHSIZE, --batchsize BATCHSIZE
                            batch size (default: 16)
      -l LR, --lr LR        learning rate (default: 0.001)
      -r RESUME, --resume RESUME
                            checkpoint number (default: 0)
      -e NUMEPOCHS, --numepochs NUMEPOCHS
                            number of epochs (default: 1000)
      -R, --doresize        whether to resize all training images to specified size (default: False)
      -s IMSIZE, --imsize IMSIZE
                            whether to resize all training images to specified size (default: 4724)
      -f BASEFOLDER, --basefolder BASEFOLDER
                            train data folder (default: datasets/OS_trees/)
    
    You should adjust the `--cellres` parameter to be suitable for your bounding box dimensions. A good guide would be to choose the power of 2 smaller than your bounding box dimension. e.g. in the paper, our tree symbols have bounding boxes of 48 pixels, so we use `--cellres 32`.

### Running inference with a model

## OS maps tree symbol dataset

We provide annotations for first edition OS map tiles. We cannot provide the image tiles themselves, however researchers are able to download these. Instead we provide only the symbol and region annotations in the format described above.

## Citation

If you use the model or the code in your research, please cite the following paper:

William A. P. Smith and Toby Pillatt. "You Only Look for a Symbol Once: An Object Detector for Symbols and Regions in Documents". In Proc. of the International Conference on Document Analysis and Recognition (ICDAR), 2023.
[https://dx.doi.org/10.1007/978-3-031-41734-4_14](https://dx.doi.org/10.1007/978-3-031-41734-4_14)

Bibtex:

    @inproceedings{smith2023yolso,
      title={You Only Look for a Symbol Once: An Object Detector for Symbols and Regions in Documents},
      author={Smith, William A. P. and Pillatt, Toby},
      booktitle={Proc. of the International Conference on Document Analysis and Recognition (ICDAR)},
      pages={227–-243},
      year={2023}
    }
