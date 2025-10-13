import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cache', default='./cache/', type=str)
parser.add_argument('--model_number', nargs='+', type=int, default=[3,3,3])
parser.add_argument('--radius', type=float, default=3)
parser.add_argument('--minpts', type=float, default=2)
args = parser.parse_args()

device = 'cuda:0'

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from stereo_toolbox.datasets import Stereo_Datasets
from stereo_toolbox.stereo_models import PSMNet
from stereo_toolbox.stereo_models import GwcNet_GC as GwcNet
from stereo_toolbox.stereo_models import PCWNet
from stereo_toolbox.disparity_regression import *


### load ensemble models [PSMNet, GwcNet, PCWNet] * 3
models = []
PSMNet_COUNT = 0
GwcNet_COUNT = 0
PCWNet_COUNT = 0

for file in os.listdir('./checkpoint/pretrained_models/'):
    if file.endswith('tar'):
        if file.startswith('PSMNet_'):
            if not PSMNet_COUNT < args.model_number[0]:
                continue
            else:
                PSMNet_COUNT += 1
            model = PSMNet()
        elif file.startswith('GwcNet_'):
            if not GwcNet_COUNT < args.model_number[1]:
                continue
            else:
                GwcNet_COUNT += 1
            model = GwcNet()
        elif file.startswith('PCWNet_'):
            if not PCWNet_COUNT < args.model_number[2]:
                continue
            else:
                PCWNet_COUNT += 1
            model = PCWNet()        

        model.disparityregression = nn.Identity()
        model = nn.DataParallel(model).to(device)
        state_dict = torch.load(f"./checkpoint/pretrained_models/{file}", map_location='cpu')['state_dict']
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        model.load_state_dict(state_dict,strict=False)
        models.append(model)

print(f'model number: {len(models)}\nmodel arch: {[x.module.__class__.__name__ for x in models]}')


class Stereo_Datasets(Stereo_Datasets):
    def __getitem__(self, index):
        left_image = self.load_image(self.datasets_list.left_images[index])
        right_image = self.load_image(self.datasets_list.right_images[index])
        disp_image = self.load_disp(self.datasets_list.disp_images[index])
        mask_image = self.load_noc_mask(self.datasets_list.disp_images[index])
        left_image, right_image, disp_image, mask_image = self.data_aug(left_image, right_image, disp_image, mask_image)
        left_image, right_image = self.processed(left_image), self.processed(right_image)
        disp_image = torch.from_numpy(np.ascontiguousarray(disp_image)).float()
        mask_image = torch.from_numpy(np.ascontiguousarray(mask_image)).float()
        return left_image, right_image, disp_image, mask_image, self.datasets_list.disp_images[index]
    

def mode_fit(mode, epsilon=1e-3):
    N, D, H, W = mode.shape
    w = mode.sum(dim=1, keepdim=True)
    mode /= w
    disp = torch.arange(D, device=mode.device).reshape(1,D,1,1)
    mu = (mode * disp).sum(dim=1,keepdim=True)
    b = (torch.abs(mu - disp) * mode).sum(dim=1,keepdim=True)

    w[w < epsilon] = 0
    mu[w < epsilon] = 0
    b[w < epsilon] = 1

    return torch.concat((w,mu,b), dim=1)


def mode_split(pred, epsilon=1e-3, max_mode_number=5):
    assert pred.dim() == 4
    M, D, H, W = pred.shape
    assert M == 1
    
    # extract mode edge
    diff = torch.diff(pred, n=1, dim=1, prepend=torch.zeros(M,1,H,W,device=pred.device))
    mode_edge = ((diff < epsilon) * torch.concat(((diff > epsilon)[:,1:,...],torch.zeros(M,1,H,W, device=pred.device, dtype=pred.dtype)), dim=1)
                + (diff < -epsilon) * torch.concat(((diff > -epsilon)[:,1:,...],torch.zeros(M,1,H,W, device=pred.device, dtype=pred.dtype)), dim=1)) > 0

    # prevent index overflow  
    mode_edge[:,[0,-1],...] = 1

    modes = []
    count = 0

    while pred.sum(dim=1).max() > epsilon and count < max_mode_number:
        indices = torch.argmax(pred, dim=1, keepdim=True).to(torch.uint8)
        mask = torch.arange(D, dtype=torch.uint8, device=pred.device).reshape([1,D,1,1]).repeat(M,1,H,W)
        mask_l = D - 1 - torch.argmax((mode_edge * (mask <= indices) * 1.0).flip(dims=[1]), dim=1, keepdim=True).to(torch.uint8)
        mask_r = torch.argmax((mode_edge * (mask > indices) * 1.0),dim=1,keepdim=True).to(torch.uint8)
        mask = (mask>=mask_l) * (mask<=mask_r)

        modes.append(mode_fit(pred * mask))
        pred = pred * ~mask

        count += 1

    return torch.concat(modes,dim=0)


def mode_clustering(modes, radius=3, minpts=2, epsilon=1e-3, max_mode_number=5):
    assert modes.dim() == 4
    assert modes.shape[1] == 3

    N, _, H, W = modes.shape
    w, mu, b = torch.split(modes, split_size_or_sections=1, dim=1)

    res = []
    
    count = 0
    while w.max() >= epsilon and count < max_mode_number:
        _, indices = torch.max(w, dim=0) 
        left = right = torch.gather(mu, dim=0, index=indices.unsqueeze(0))  
        left, right = left - radius, right + radius
        mask = (mu >= left) * (mu <= right)

        while True:
            left, right = torch.min(mu * mask + 192 * ~mask, dim = 0, keepdim=True)[0] - radius, torch.max(mu * mask, dim = 0, keepdim=True)[0] + radius

            mask_new = (mu >= left) * (mu <= right)

            if torch.equal(mask, mask_new):
                break
            else:
                mask = mask_new

        fusion_mode = (modes * mask).mean(dim=0) * N / mask.sum(dim=0)
        fusion_mode[:, mask.sum(dim=0).squeeze(0) < minpts] = torch.tensor([0,0,1], dtype=fusion_mode.dtype, device=fusion_mode.device)[:, None]

        res.append(fusion_mode)

        w[mask] = 0

        count += 1

    res = torch.stack(res, dim=0)

    return res


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
    w_mask = modes[0,0,...] > modes[1:,0,...].sum(0)
    modes[0,0,...] = modes[0,0,...] * w_mask + modes[1:,0,...].sum(0) * ~w_mask
    modes[0,1,...] = disp.squeeze()
    modes[0,2,...] = (modes[0,2,...] - modes[0,2,...].mean() + (min_b+max_b)/2).clamp(min=min_b, max=max_b)
    return modes
    

TrainImgLodader = DataLoader(Stereo_Datasets(('sceneflow', 'train_finalpass'), training=False),  
                                   batch_size=1, shuffle=False, num_workers=8, drop_last=False)


for _, (left, right, disp, noc_mask, disp_file_name) in enumerate(tqdm(TrainImgLodader)):
    left, right, disp = left.to(device), right.to(device), disp.to(device)

    h, w = np.random.randint(0,left.shape[-2]), np.random.randint(0,left.shape[-1])

    modes = []
    for model in models:
        with torch.no_grad():
            output = model.eval()(left, right).to(device)
            modes.append(mode_split(output))

    modes = torch.concat(modes, dim=0)  # M*D*H*W

    gt_b = 0.8
    gt_mode = torch.stack((torch.ones_like(disp), disp, torch.ones_like(disp) * gt_b), dim = 1).repeat(3,1,1,1)
    modes = torch.concat((modes,gt_mode), dim=0)

    modes = mode_clustering(modes, radius=args.radius, minpts=args.minpts, max_mode_number=5)
    # modes = mode_correction(modes, disp.to(modes.device))

    modes = modes[...,-540:,:].cpu().numpy().astype(np.float16)

    save_file_name = disp_file_name[0].replace('/data/xp/Scene_Flow/', f'{args.cache}').replace('.pfm', '.npz')
    os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
    np.savez_compressed(save_file_name, arr=modes)


