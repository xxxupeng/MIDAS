import argparse
parser = argparse.ArgumentParser('Stereo Matching', add_help=False)
parser.add_argument('--cache', default='./cache/', type=str)
parser.add_argument('--mode_correction', default=True, type=bool)
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--model', default='PSMNet', type=str,choices=['PSMNet','GwcNet_GC','PCWNet_GC'])
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
from PIL import Image
import random

from stereo_toolbox.datasets import Stereo_Datasets
from stereo_toolbox.datasets.data_augmentation import Data_Augmentation
from stereo_toolbox.stereo_models import PSMNet
from stereo_toolbox.stereo_models import GwcNet_GC as GwcNet
from stereo_toolbox.stereo_models import PCWNet
from stereo_toolbox.disparity_regression import *
from stereo_toolbox.evaluator import Evaluator_Toolbox


class Data_Augmentation(Data_Augmentation):
    def random_crop(self, left, right, disp=None, mask=None, distribution=None):
        left = np.array(left)[:504]
        right = np.array(right)[:504]

        H, W, C = left.shape
        crop_H, crop_W = self.crop_size
        if crop_H > H: crop_H = H
        if crop_W > W: crop_W = W

        h = random.randint(0, H - crop_H)
        w = random.randint(0, W - crop_W)

        left = left[h : h + crop_H, w : w + crop_W]
        right = right[h : h + crop_H, w : w + crop_W]
        if disp is not None:
            disp = disp[...,:504,:]
            disp = disp[..., h : h + crop_H, w : w + crop_W]
        if mask is not None:
            mask = mask[h : h + crop_H, w : w + crop_W]
        if distribution is not None:
            distribution = distribution[...,-504:,:]
            distribution = distribution[..., h : h + crop_H, w : w + crop_W]

        return left, right, disp, mask, distribution
    
    def __call__(self, left, right, disp, mask, distribution):            
        raw_left, raw_right, disp, mask, distribution = self.random_crop(left, right, disp, mask, distribution)
        left = Image.fromarray(raw_left)
        right = Image.fromarray(raw_right)
        left, right = self.random_jitter(left, right)
        right = self.random_mask(right)

        return left, right, disp, mask, distribution


class Stereo_Datasets(Stereo_Datasets):
    def __init__(self, *datasets: tuple, training: bool):
        super().__init__(*datasets, training=training)
        self.data_aug = Data_Augmentation(self.training)

    def load_distribution(self, filename: str):
        return np.load(filename)['arr']
    
    def __getitem__(self, index):
        left_image = self.load_image(self.datasets_list.left_images[index])
        right_image = self.load_image(self.datasets_list.right_images[index])
        disp_image = self.load_disp(self.datasets_list.disp_images[index])
        distribution = self.load_distribution(self.datasets_list.disp_images[index].replace('/data/xp/Scene_Flow', f'{args.cache}').replace('.pfm','.npz'))
        
        left_image, right_image, disp_image, _, distribution = self.data_aug(left_image, right_image, disp_image, None, distribution)
        left_image = self.processed(left_image)
        right_image = self.processed(right_image)
        
        disp_image = np.ascontiguousarray(disp_image, dtype=np.float32)
        disp_image = torch.from_numpy(disp_image).float()

        distribution = np.ascontiguousarray(distribution, dtype=np.float32)
        distribution = torch.from_numpy(distribution).float()

        return left_image, right_image, disp_image, distribution


def mode_modeling(modes, maxdisp=192):
    if modes.dim() == 4:
        assert modes.shape[1] == 3
        modes = modes[None,...]
    elif modes.dim() == 5:
        assert modes.shape[2] == 3
    else:
        assert 0

    B, N, _, H, W = modes.shape

    disp = torch.arange(maxdisp, device=modes.device)[None, None,...,None,None]
    res = (F.softmax(-torch.abs(disp - modes[:,:,1:2,...]) / modes[:,:,2:3,...].clamp(min=1e-3), dim=2) * modes[:,:,0:1,...] ).sum(dim=1)

    return res / res.sum(dim=1, keepdim=True).clamp(min=1e-3,max=1e3)


def mode_correction(modes, disp, min_b=0.6, max_b=1.0):
    if modes.ndim == 4:
        modes = modes[None,...]

    w_mask = modes[:,0,0,...] > modes[:,1:,0,...].sum(1)
    modes[:,0,0,...] = modes[:,0,0,...] * w_mask + modes[:,1:,0,...].sum(1) * ~w_mask

    modes[:,0,1,...] = disp.squeeze()

    modes[:,0,2,...] = (modes[:,0,2,...] - modes[:,0,2,...].mean() + (min_b+max_b)/2).clamp(min=min_b, max=max_b)
    return modes
    

train_domains = [('sceneflow', 'train_finalpass')]

dataset = Stereo_Datasets(*train_domains, training=True)
TrainImgLoader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)

ce_loss = lambda input, label, mask: - (label[mask]*(torch.log(input + 1e-30))[mask]).sum() / (mask.sum() / 192)

def train_iter(model_s, device_s, optimizer, scheduler, left, right, disp, distribution):
    left, right, disp, distribution = left.to(device_s), right.to(device_s), disp.to(device_s), distribution.to(device_s)

    if args.mode_correction:
        distribution = mode_modeling(mode_correction(distribution, disp)).detach_()
    else:
        distribution = mode_modeling(distribution).detach_()

    # mask
    mask = ((disp > 0) * (disp <= 191) * torch.isclose(distribution.sum(dim=1), torch.tensor(1.0), atol=1e-3)).detach_()
    if mask.sum() == 0:
        return float(0)
    mask = mask.unsqueeze(1).expand_as(distribution)

    ## training
    model_s.train()
    optimizer.zero_grad()
    output = model_s(left, right)

    if type(model_s.module).__name__ == 'PSMNet':
        loss = (0.5 * ce_loss(output[0], distribution, mask)
                + 0.7 * ce_loss(output[1], distribution, mask)
                + 1.0 * ce_loss(output[2], distribution, mask)
        )
    elif type(model_s.module).__name__ == 'GwcNet':
        loss = (0.5 * ce_loss(output[0], distribution, mask)
                + 0.5 * ce_loss(output[1], distribution, mask)
                + 0.7 * ce_loss(output[2], distribution, mask)
                + 1.0 * ce_loss(output[3], distribution, mask)
        )
    elif type(model_s.module).__name__ == 'PWCNet':
        loss = (0.5 * ce_loss(output[0], distribution, mask)
                + 0.5 * ce_loss(output[1], distribution, mask)
                + 0.5 * ce_loss(output[2], distribution, mask)
                + 0.7 * ce_loss(output[3], distribution, mask)
                + 1.0 * ce_loss(output[4], distribution, mask)
        )

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def main(saved_dir):
    device_s = "cuda:0"
    if args.model == 'PSMNet':
        model_s = PSMNet()
    elif args.model == 'GwcNet_GC':
        model_s = GwcNet()
    elif args.model == 'PCWNet_GC':
        model_s = PCWNet()
    
    model_s.disparityregression = nn.Identity()
    model_s = nn.DataParallel(model_s, device_ids=[int(device_s[5])]).to(device_s)

    total_epochs = 80
    optimizer = optim.Adam(model_s.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            max_lr=1e-3,
                                            epochs=total_epochs,
                                            steps_per_epoch=len(TrainImgLoader),
                                            )

    for epoch in range(total_epochs):
        loss = 0

        for _, (left, right, disp,  distribution) in enumerate(tqdm(TrainImgLoader)):
            loss += train_iter(model_s, device_s, optimizer, scheduler, left, right, disp, distribution)

        saved_name = os.path.join(saved_dir, f"epoch_{epoch:02d}.tar")
        os.makedirs(os.path.dirname(saved_name), exist_ok=True)
        torch.save({
            'state_dict_student': model_s.state_dict(),
        }, saved_name)

        # evaluator = Evaluator_Toolbox(model_s, device_s, disparityregression = unimodal_disparityregression_SA())
        # epe_3px = evaluator.eval(filelist='test_finalpass')


if __name__ == '__main__':
    saved_dir = f"./checkpoint/{args.model}/"
    os.makedirs(saved_dir, exist_ok=True)

    main(saved_dir)
