import random
from torchvision.transforms import functional as f

class AdjustGamma(object):
    def __init__(self, gamma, center=1):
        self.gamma = random.uniform(center - gamma, center + gamma)

    def __call__(self, sample):
        return f.adjust_gamma(sample, self.gamma)