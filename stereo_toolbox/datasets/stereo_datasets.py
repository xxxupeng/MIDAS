import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from .datasets_lists import Datasets_List
from .data_augmentation import Data_Augmentation
from .utils import *

class Stereo_Datasets(Dataset):
    def __init__(self, *datasets: tuple, training: bool):
        """
        datasets: (dataset1, mode1), (dataset1, mode2), (dataset2, mode3),...
        """
        self.datasets_list = Datasets_List(*datasets)
        self.training = training
        self.data_aug = Data_Augmentation(self.training)
        self.processed = get_transform()

        
    def __len__(self):
        return len(self.datasets_list.left_images)


    def load_image(self, filename: str):
        return Image.open(filename).convert('RGB')


    def load_disp(self, filename: str):
        if filename is None:
            return None
        
        if filename.endswith('.png'):
            disp = Image.open(filename)
            disp = np.array(disp, dtype=np.float32) / 256.
        elif filename.endswith('.pfm'):
            disp, _ = pfm_imread(filename)
            disp = np.ascontiguousarray(disp, dtype=np.float32)
        elif filename.endswith('.npy'):
            disp = np.load(filename)
            disp = np.ascontiguousarray(disp, dtype=np.float32)
        return disp
    

    def load_noc_mask(self, filename: str):
        if 'Middlebury' in filename or 'ETH3D' in filename:
            filename = filename.replace('disp0GT.pfm', 'mask0nocc.png')
            # data = plt.imread(filename)
            # data = (data == 1)
            # return data
            noc_mask = Image.open(filename).convert('L')
            noc_mask = np.array(noc_mask, dtype=np.uint8)
            return noc_mask == 255
        else:
            return np.ones_like(self.load_disp(filename))


    def __getitem__(self, index):
        self.index = index
        left_image = self.load_image(self.datasets_list.left_images[index])
        right_image = self.load_image(self.datasets_list.right_images[index])
        disp_image = self.load_disp(self.datasets_list.disp_images[index])
        mask_image = self.load_noc_mask(self.datasets_list.disp_images[index])
        
        if self.training:
            left_image, right_image, disp_image, mask_image, raw_left, raw_right = self.data_aug(left_image, right_image, disp_image, mask_image)
            # raw_left = self.processed(raw_left)
            # raw_right = self.processed(raw_right)
            raw_left = transforms.ToTensor()(raw_left)
            raw_right = transforms.ToTensor()(raw_right)
        else:
            left_image, right_image, disp_image, mask_image = self.data_aug(left_image, right_image, disp_image, mask_image)

        left_image = self.processed(left_image)
        right_image = self.processed(right_image)
        
        if disp_image is not None:
            disp_image = np.ascontiguousarray(disp_image)
            disp_image = torch.from_numpy(disp_image).float()

            mask_image = np.ascontiguousarray(mask_image)
            mask_image = torch.from_numpy(mask_image).float()

            if self.training:
                return left_image, right_image, disp_image, mask_image, raw_left, raw_right
            else:
                return left_image, right_image, disp_image, mask_image
        
        else:
            return left_image, right_image



