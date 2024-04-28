import pickle
from typing import Tuple
import numpy as np
import os
from PIL import Image

from dlvc.datasets.dataset import Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.data = self.load_cifar()

    def unpickle(self, file: str):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_cifar(self):
        files = []
        if self.subset == Subset.TRAINING:
            for i in range(1, 5):
                files.append(os.path.join(self.fdir, "data_batch_" + str(i)))

        elif self.subset == Subset.VALIDATION:
            files.append(os.path.join(self.fdir, "data_batch_5"))

        elif self.subset == Subset.TEST:
            files.append(os.path.join(self.fdir, "test_batch"))

        image_list = []
        label_list = []

        for f in files:

            data = self.unpickle(f)
            img_data = data[b"data"]
            img_data = np.reshape(
                img_data, (img_data.shape[0], 32, 32, 3), order="F")

            img_data = img_data.transpose(0, 2, 1, 3)

            image_list.append(img_data)

            label_list.append(data[b"labels"])

        image_arr = np.concatenate(image_list)
        label_arr = np.concatenate(label_list)

        return (image_arr, label_arr)

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return self.data[1].shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        if idx < 0 or idx >= len(self.data[0]):
            raise IndexError("Index is out of bounds")

        image = self.data[0][idx]
        label = self.data[1][idx]


        if self.transform != None:
            # needs to be done otherwise transform does not apply directly on numpy arrays
            image = Image.fromarray(image, mode='RGB')
            image = self.transform(image)
            image = np.array(image)

        return (image, label)

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return len(self.classes)
