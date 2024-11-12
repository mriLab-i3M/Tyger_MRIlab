import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch


class individualDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, dataA, dataB):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.input_nc = 1
        self.output_nc = 1
        self.model_shape = opt.model_shape
        self.model_augmentation = opt.model_augmentation
        self.A_paths = [1]
        self.B_paths = [1]
        self.dataA = dataA
        self.dataB = dataB

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            B (tensor) - - an image in the input domain
            A (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        Tengo que seleccionar dirección BtoA!!!!
        """
        A = self.dataA
        B = self.dataB
        min_range = -1
        max_range = 1
        model_shape = np.array([self.model_shape,self.model_shape])
        out_shape = np.array([1, self.model_shape,self.model_shape])

        def normalization(img, min_range, max_range):
            imgNorm = (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_range - min_range) + min_range
            return imgNorm

        def downSampling(img, down_shape):
            kSpace = np.fft.fftshift(np.fft.fftn(img))
            ini_shape = np.array(kSpace.shape)
            center = (ini_shape / 2).astype(int) - (down_shape / 2).astype(int)
            kSpaceDownSamp = kSpace[center[0]:center[0] + down_shape[0], center[1]:center[1] + down_shape[1]]
            imgDown = np.abs(np.fft.ifftn((kSpaceDownSamp)))
            return imgDown
        
        def zeroPadding(img, up_shape):
            kSpace = np.fft.fftshift(np.fft.fftn(img))
            ini_shape = np.array(kSpace.shape)
            center = (up_shape / 2).astype(int) - (ini_shape / 2).astype(int)
            kSpaceUpSamp = np.zeros(up_shape).astype(complex)
            kSpaceUpSamp[center[0]:center[0] + ini_shape[0], center[1]:center[1] + ini_shape[1]] = kSpace
            imgUp = np.abs(np.fft.ifftn((kSpaceUpSamp)))
            return imgUp
    
        if self.model_shape < A.shape[0]:
            A = downSampling(A, model_shape)
        else:
            A = zeroPadding(A, model_shape)
        
        if self.model_shape < B.shape[0]:
            B = downSampling(B, model_shape)
        else:
            B = zeroPadding(B, model_shape)

        B = normalization(B, min_range, max_range)
        A = normalization(A, min_range, max_range)

        A = torch.tensor(np.reshape(A, out_shape), dtype=torch.float32)
        B = torch.tensor(np.reshape(B, out_shape), dtype=torch.float32)
        return {'A': A, 'B': B, 'A_paths': 'none', 'B_paths': 'none'} 

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
