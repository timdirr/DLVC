# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:50:59 2024

@author: ericx
"""

    
import os
import glob
import pickle
import numpy as np
from matplotlib import pyplot as plt

from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Tuple
import torchvision.transforms.v2 as v2
import torch
from PIL import Image
from dlvc.datasets.cifar10 import  CIFAR10Dataset
from dlvc.datasets.dataset import  Subset


ROOT_DIR = os.getcwd()

fdir    = os.path.join(ROOT_DIR,"cifar-10-batches-py")
# subset  = Subset.TRAINING  
# subset  = Subset.VALIDATION 
subset  = Subset.TEST      
 

index = 1

#https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html

transform = v2.Compose([
    v2.RandomHorizontalFlip(p=1),
])
   
dataset = CIFAR10Dataset(fdir = fdir, subset = subset, transform=None) 

img     = dataset[index][0]
label   = dataset[index][1] 
    
print(f"dataset length: {len(dataset)}")
print(f"number of classes: {dataset.num_classes()}")

plt.close('all')

plt.figure()
plt.imshow(img)
plt.title(label)

dataset.transform = transform

img     = dataset[index][0]
label   = dataset[index][1]

plt.figure()
plt.imshow(img)
plt.title(label + " flipped")


    
#%%





