import torch
import torch.nn as nn
import torch.nn.functional as F


def SSIM(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1) 

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2) 

    base_grid = torch.stack([x_base, y_base], 1)
    return base_grid

def disp_warp(x, disp, r2l=False, pad='zeros', mode='bilinear', device='cuda'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)
    v_grid = norm_grid(base_grid + torch.cat((offset*disp,torch.zeros_like(disp)),1)) 
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    mask = torch.autograd.Variable(torch.ones(x_recons.size())).to(device)
    mask = nn.functional.grid_sample(mask, v_grid)
    return x_recons, mask


def photometric_loss(im1_scaled, im1_recons):
    loss = []
    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])


def binocular_photometric_loss(disp, im1, im2):
    if len(disp.shape) == 3:
        disp = disp.unsqueeze(1)
    elif len(disp.shape) == 4:
        assert disp.shape[1] == 1, "First dimension must be 1 for 4D disp tensor"
        
    im1_recons, _ = disp_warp(im2, disp, r2l=False)

    loss_warp = photometric_loss(im1, im1_recons).squeeze()
    loss_2 = photometric_loss(im2, im1).squeeze()

    automask = loss_warp < loss_2
    loss = (loss_warp)[automask]

    return loss.mean()