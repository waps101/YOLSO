# [You Only Look for a Symbol Once: An Object Detector for Symbols and Regions in Documents]([http://openaccess.thecvf.com/content_CVPR_2020/papers/Smith_A_Morphable_Face_Albedo_Model_CVPR_2020_paper.pdf](https://eprints.whiterose.ac.uk/198859/1/_ICDAR_YOLSO.pdf))

 [William A. P. Smith](https://www-users.cs.york.ac.uk/wsmith) and [Toby Pillatt]([https://www.linkedin.com/in/alassane-seck-67508365](https://www.york.ac.uk/archaeology/people/toby-pillatt/))
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

### Training a model

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
