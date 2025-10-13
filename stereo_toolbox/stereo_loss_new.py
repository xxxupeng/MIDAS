import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_right_to_left(right_image, disp):
    """将右图像根据视差图重投影到左图像视角
    
    使用给定的视差图将右图像变形到左图像的视角，实现视图合成。
    这是自监督立体匹配中计算重建损失的基础操作。
    
    参数:
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        
    返回:
        Tensor: 重投影到左视角的右图像
    """
    batch_size, _, height, width = right_image.size()
    
    # 生成网格坐标
    device = disp.device
    x_base = torch.linspace(0, 1, width, device=device).repeat(batch_size, height, 1)
    y_base = torch.linspace(0, 1, height, device=device).repeat(batch_size, width, 1).transpose(1, 2)
    flow_field = torch.stack((x_base - disp.squeeze(1) / (width-1), y_base), dim=3)
    
    # 使用grid_sample进行重投影
    warped_right = F.grid_sample(right_image, 
                                 (flow_field * 2 - 1),  # 转换到[-1,1]范围
                                 mode='bilinear', 
                                 padding_mode='zeros')
    
    return warped_right


def ssim(x, y, window_size=7, pad_mode='reflect'):
    """计算结构相似性(SSIM)损失
    
    SSIM考虑图像结构信息，比简单L1损失对光照变化更鲁棒。
    返回的是SSIM距离损失：(1-SSIM)/2，范围为[0,1]，值越小表示图像越相似。
    
    参数:
        x (Tensor): 第一个图像，形状为 [B, C, H, W]
        y (Tensor): 第二个图像，形状为 [B, C, H, W]
        window_size (int): SSIM计算的窗口大小
        
    返回:
        Tensor: SSIM距离损失，形状为 [B, C, H, W]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 使用反射填充
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    y_padded = F.pad(y, (pad, pad, pad, pad), mode=pad_mode)
    
    # 无填充的平均池化（因为已经填充）
    mu_x = F.avg_pool2d(x_padded, window_size, stride=1)
    mu_y = F.avg_pool2d(y_padded, window_size, stride=1)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.avg_pool2d(x_padded * x_padded, window_size, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y_padded * y_padded, window_size, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x_padded * y_padded, window_size, stride=1) - mu_xy
    
    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)


def photometric_loss(left_image, right_image, disp=None, ssim_weight=0.85):
    """计算光度一致性损失
    
    结合SSIM损失和L1损失的加权和，用于评估视差预测的准确性。
    SSIM损失捕获结构信息，L1损失捕获细节信息。
    
    参数:
        left_image (Tensor): 左图像，形状为 [B, C, H, W]
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        ssim_weight (float): SSIM损失的权重，默认为0.85
        
    返回:
        Tensor: 光度一致性损失，形状为 [B, C, H, W]
    """
    if disp is None:
        warped_right_image = right_image
    else:
        warped_right_image = warp_right_to_left(right_image, disp)

    # 计算光度一致性损失
    return ssim_weight * ssim(left_image, warped_right_image).mean(1, True) + (1-ssim_weight) * torch.abs(left_image - warped_right_image).mean(1, True)


def edge_aware_smoothness_loss(disp, img):
    """计算边缘感知的平滑度损失
    
    鼓励视差图在图像边缘处不连续，在平滑区域保持平滑。
    通过图像梯度调制视差梯度损失，使视差边界与图像边界对齐。
    
    参数:
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        img (Tensor): 输入图像，形状为 [B, C, H, W]
        
    返回:
        Tensor: 边缘感知的平滑度损失
    """
    # 确保输入图像已归一化到0-1范围
    if img.max() > 1.0:
        print("Warning: Image may not be normalized. Expected range: [0,1]")
    
    # 计算视差梯度
    disp_dx = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    disp_dy = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    
    # 计算图像梯度 (对归一化后的图像)
    img_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    
    # 边缘感知权重（图像梯度处权重较小）
    weights_x = torch.exp(-img_dx)
    weights_y = torch.exp(-img_dy)
    
    # 应用权重
    smoothness_x = disp_dx * weights_x
    smoothness_y = disp_dy * weights_y

    loss = torch.mean(smoothness_x) + torch.mean(smoothness_y)
    
    return loss


# def lr_consistency_loss(left_disp, right_disp):
#     """计算左右视差一致性损失
    
#     强制左右视差图之间的几何一致性，有助于处理遮挡区域和改善整体一致性。
#     通过将左视差映射到右视角，并与右视差比较来实现。
    
#     参数:
#         left_disp (Tensor): 左视差图，形状为 [B, 1, H, W]
#         right_disp (Tensor): 右视差图，形状为 [B, 1, H, W]
        
#     返回:
#         Tensor: 左右一致性损失，形状为 [B, 1, H, W]
#     """
#     batch_size, _, height, width = left_disp.size()
    
#     # 生成坐标网格
#     device = left_disp.device
#     x_base = torch.linspace(0, width-1, width, device=device).repeat(batch_size, height, 1)
    
#     # 根据左视差图找到右图中对应点
#     x_left_to_right = x_base - left_disp
    
#     # 将坐标归一化到[-1,1]
#     norm_x_coords = (2 * x_left_to_right / (width - 1)) - 1
#     norm_y_coords = torch.linspace(-1, 1, height, device=device).repeat(batch_size, width, 1).transpose(1, 2)
#     grid = torch.stack((norm_x_coords, norm_y_coords), dim=3)
    
#     # 采样右视差图
#     right_disp_at_left = F.grid_sample(right_disp, grid, mode='bilinear', padding_mode='border')
    
#     # 计算一致性损失
#     lr_loss = torch.abs(left_disp - right_disp_at_left)
    
#     return lr_loss


def auto_mask_loss(loss, left_image, right_image, disp):
    """自动掩码损失函数，处理自监督立体匹配中的难以匹配区域
    
    该函数实现了自动掩码机制，通过比较基于预测视差的重投影误差与直接图像匹配误差，
    识别并排除以下问题区域：
    1. 远距离/无穷远区域（如天空）：视差接近零，难以准确估计
    2. 低纹理区域：多种视差值可能产生相似的重投影结果
    3. 非共视区域：在左图可见但右图中不存在对应点的区域
    
    原理：当预测视差产生的重投影误差不小于直接使用未变换右图的误差时，
    说明该区域可能是上述问题区域之一，应排除在训练中，避免引入噪声。
    
    参数:
        loss (Tensor): 原始像素级损失，形状为 [B, 1, H, W]
        left_image (Tensor): 左图像，形状为 [B, C, H, W]
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        
    返回:
        Tensor: 应用自动掩码后的损失
    """
    # 计算重投影误差和直接比较误差
    reproj_error = photometric_loss(left_image, right_image, disp.detach())
    identity_error = photometric_loss(left_image, right_image)
    
    # 创建掩码 - 只在重投影有效的区域应用损失
    mask = (reproj_error < identity_error).float()
    
    # 确保掩码不会全为零导致梯度消失
    valid_pixels = torch.sum(mask) + 1e-8
    
    # 应用掩码并计算平均损失
    loss = torch.sum(loss * mask) / valid_pixels
    
    return loss


def min_percent_loss(loss, percent=20):
    """选择损失图中最小的百分之x进行计算的损失函数
    
    该函数实现了基于损失值筛选的机制，仅保留损失值最小的前x%像素参与最终损失计算。
    这种方法基于以下假设：
    1. 损失值较小的像素更可能是准确匹配的区域
    2. 损失值较大的像素可能来自遮挡区域、非共视区域或匹配困难区域
    3. 通过逐步聚焦于可靠区域，网络能学习到更准确的视差估计
    
    原理：对损失图进行排序，仅保留前x%最小值参与计算，其余部分被掩码排除，
    这可以防止异常值和困难区域主导训练过程，提高模型鲁棒性。
    
    参数:
        loss (Tensor): 原始像素级损失，形状为 [B, 1, H, W]
        percent (int): 要保留的最小损失像素的百分比，取值范围[0,100]
        
    返回:
        Tensor: 仅包含最小x%损失值的平均损失
    """
    batch_size, _, height, width = loss.shape
    num_pixels = height * width
    k = int(num_pixels * percent / 100)
    
    # 将每个批次的损失展平并按值排序
    loss_flat = loss.view(batch_size, -1)
    
    # 获取每个批次中最小的k个值
    top_values, _ = torch.topk(loss_flat, k, dim=1, largest=False)
        
    # 取每个批次k个最小值的均值
    valid_mask = (top_values < 1e4).float()
    batch_means = (top_values * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    
    # 取所有批次的均值
    loss = batch_means.mean()
    return loss


def auto_mask_min_percent_loss(loss, left_image, right_image, disp, percent=20):
    """结合自动掩码和最小百分比损失的损失函数
    
    该函数结合了自动掩码和最小百分比损失的优点，通过自动掩码排除难以匹配区域，
    再从剩余区域中选择最小损失的前x%像素参与损失计算，以提高模型训练的稳定性和准确性。
    
    参数:
        loss (Tensor): 原始像素级损失，形状为 [B, 1, H, W]
        left_image (Tensor): 左图像，形状为 [B, C, H, W]
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        percent (int): 要保留的最小损失像素的百分比，取值范围[0,100]
        
    返回:
        Tensor: 应用自动掩码和最小百分比损失后的损失
    """
    # 计算重投影误差和直接比较误差
    reproj_error = photometric_loss(left_image, right_image, disp.detach())
    identity_error = photometric_loss(left_image, right_image)
    
    # 创建无效区域掩码 - 只在重投影有效的区域应用损失 
    # 当重投影误差小于直接比较误差时，这些点是有效的
    mask = (reproj_error > (identity_error - 1e-7)).float()
    
    # 应用掩码，无效区域部分加上10086
    masked_loss = loss + mask * 10086

    # 计算最小百分比损失
    loss = min_percent_loss(masked_loss, percent)
    return loss
