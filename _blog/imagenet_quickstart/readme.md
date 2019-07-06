---
layout: notes_without_title
section-type: notes
title: imagenet quickstart
category: blog
---



# Quick start imagenet in pytorch

**chandan singh** 

---

[ImageNet](http://www.image-net.org/) has become a staple dataset in computer vision, but is still pretty difficult to download/install. These are some simple instructions to get up and running in pytorch.

## step 1: download/preprocessing

- begin by following the instructions for downloading the ImageNet dataset [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)
- the dataset contains ~1.2 million training images and 50,000 validation images
- note that the dataset is quite large: the .tar files are 138G for train and 6.3G for val
  - once extracted the train data is 177G and the val data is 8.3G
- the folder hierarchy looks liks this (val looks similar, there are 1000 folders, one for each class):
```python
train/
	n04550184/
		n04550184_9946.JPEG
        n04550184_9945.JPEG  
        ...
	n04550180/
	...
	n04550180/
```

## step 2: get the names for each class

- to get the names for each of the classes, look at the [class_names_imagenet.py](class_names_imagenet.py) file, which contains a dictionary containing the class labels

## step 3: set up a dataloader

- in pytorch, the dataloader should be set up using an [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet-12)
- there is an example [here](https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126)
