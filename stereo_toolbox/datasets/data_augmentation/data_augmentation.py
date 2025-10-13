from torchvision.transforms import ColorJitter, functional, Compose
import numpy as np
import random
from PIL import Image

from .utils import *

class Data_Augmentation():
    def __init__(self, training: bool, crop_size = [256, 512]):
        self.training = training

        self.color_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14), 
            AdjustGamma(0.2)])
        
        self.crop_size = crop_size


    def random_jitter(self, left, right):
        left = np.array(self.color_aug(left))
        right = np.array(self.color_aug(right))

        return left, right


    def random_crop(self, left, right, disp=None, mask=None):
        left = np.array(left)
        right = np.array(right)

        H, W, C = left.shape
        crop_H, crop_W = self.crop_size
        if crop_H > H: crop_H = H
        if crop_W > W: crop_W = W

        h = random.randint(0, H - crop_H)
        w = random.randint(0, W - crop_W)

        left = left[h : h + crop_H, w : w + crop_W]
        right = right[h : h + crop_H, w : w + crop_W]
        if disp is not None:
            disp = disp[..., h : h + crop_H, w : w + crop_W]
        if mask is not None:
            mask = mask[h : h + crop_H, w : w + crop_W]

        return left, right, disp, mask


    def random_mask(self, right):
        right.flags.writeable = True
        if np.random.binomial(1, 0.5):
            sx = int(np.random.uniform(35,100))
            sy = int(np.random.uniform(25,75))
            cx = int(np.random.uniform(sx, right.shape[0]-sx))
            cy = int(np.random.uniform(sy, right.shape[1]-sy))
            right[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right,0),0)[np.newaxis,np.newaxis]
        
        return right

    def pad_to_2x(self, left, right, disp=None, mask=None):
        left = np.array(left)
        right = np.array(right)

        H, W, C = left.shape

        scale = 96
        top_pad = int(np.ceil(H / scale) * scale - H)
        right_pad = int(np.ceil(W / scale) * scale - W)

        left = np.lib.pad(left, ((top_pad, 0), (0, right_pad), (0, 0)), 
                              mode='constant', constant_values=0)
        right = np.lib.pad(right, ((top_pad, 0), (0, right_pad), (0, 0)), 
                               mode='constant', constant_values=0)
        
        if disp is not None:
            if len(disp.shape) == 2:
                disp = np.lib.pad(disp, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            elif len(disp.shape) == 3:
                disp = np.lib.pad(disp, ((0,0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            
        if mask is not None:
            assert len(mask.shape) == 2
            mask = np.lib.pad(mask, ((top_pad, 0), (0, right_pad)),
                                   mode='constant', constant_values=0)
            

        return left, right, disp, mask


    def __call__(self, left, right, disp, mask):
        if self.training:
            
            raw_left, raw_right, disp, mask = self.random_crop(left, right, disp, mask)
            left = Image.fromarray(raw_left)
            right = Image.fromarray(raw_right)
            left, right = self.random_jitter(left, right)
            right = self.random_mask(right)

            return left, right, disp, mask, raw_left, raw_right

        else:
            left, right, disp, mask = self.pad_to_2x(left, right, disp, mask)
            
            return left, right, disp, mask


        