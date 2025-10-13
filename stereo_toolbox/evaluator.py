import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

try:
    from .datasets import Stereo_Datasets
    from .disparity_regression import *
except:
    from datasets import Stereo_Datasets
    from disparity_regression import *


class Evaluator_Toolbox():
    def __init__(self, model=None, device="cuda:0", threshold=3, disparityregression = nn.Identity()):
        self.model = model
        self.device = device
        self.threshold = threshold

        if type(self.model.module).__name__ == "GANet":
            disparityregression.maxdisp = disparityregression.maxdisp + 1
            self.model.module.cost_agg.disp2.disparityregression = disparityregression
        else:
            self.disparityregression = disparityregression


    def eval_iter(self, left, right, disp, noc_mask):
        left, right, disp = left.to(self.device), right.to(self.device), disp.to(self.device)
        
        mask = ((disp > 0) * (disp <= 191)).detach_()
        if noc_mask is not None:
            noc_mask = noc_mask.to(self.device)
            mask = mask * noc_mask
        mask = mask.bool()

        if mask.sum() == 0:
            return float(0), float(0)
        
        self.model.eval()

        with torch.no_grad():
            output = self.model(left, right)
        
        if output.shape[1] == 1:
            output = output.squeeze(1)
        else:
            output = self.disparityregression(output).squeeze(1)

        epe = (output[mask] - disp[mask]).abs().mean()
        outliers = ((output[mask] - disp[mask]).abs() > self.threshold).sum()  * 100 / mask.sum()

        return epe.item(), outliers.item()
    

    def eval(self, dataset='sceneflow', filelist='test_finalpass', threshold=None, mask_flag=True):        
        if threshold is not None:
            self.threshold = threshold

        TestImgLoader = DataLoader(Stereo_Datasets((dataset, filelist), training=False),  
                                   batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        epe = outliers = 0
        for _, (left, right, disp, noc_mask) in enumerate(TestImgLoader):
            noc_mask = noc_mask if mask_flag else None
            epe, outliers = map(sum, zip((epe, outliers), self.eval_iter(left, right, disp, noc_mask)))

        self.epe = epe / len(TestImgLoader)
        self.outliers = outliers / len(TestImgLoader)

        print(f"{dataset}:  EPE {self.epe:.6f}px,  Outliers({self.threshold:1d}px) {self.outliers:.6f}%.")

        return self.epe, self.outliers

    def eval_generalization(self):
        thresholds = {'kitti2015':3, 'kitti2012':3, 'middlebury':2, 'eth3d':1}
        test_domains = [
                ('kitti2015', 'train_all'),
                ('kitti2012', 'train_all'),
                ('middlebury', 'train_all'),
                ('eth3d', 'train_all')
                ]
        
        self.generalization = []
        for _, (dataset, filelist) in enumerate(test_domains):
            epe, outliers = self.eval(dataset, filelist, thresholds[dataset], True)
            self.generalization.append([dataset, epe, outliers])

        return np.array([item[1:] for item in self.generalization])



if __name__ == '__main__':
    from stereo_models import PSMNet
    from disparity_regression import *

    device = "cuda:1"
    model = PSMNet()
    model.disparityregression = unimodal_disparityregression_SA()
    model = nn.DataParallel(model, device_ids=[int(device[5])]).to(device)

    evaluator = Evaluator_Toolbox(model, device)

    for i in range(23,50,1):
        print(f"epoch {i:02d}")
        state_dict = torch.load(f"/data/xp/Check_Point/distill/SceneFlow/pseudo+gt_epoch_{i:02d}.tar")['state_dict']
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        model.load_state_dict(state_dict,strict=False)

        evaluator.eval('sceneflow','test_finalpass')
        print('-' * 80)
