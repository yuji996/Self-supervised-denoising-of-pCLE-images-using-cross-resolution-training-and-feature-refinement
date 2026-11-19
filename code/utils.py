import torch
import numpy as np
import math
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import os
import torch
import torch.nn.functional as F

class Sub_sampler():
    def __init__(self, scale_factor, device):
        self.scale = scale_factor
        self.device = device

    def sample_img(self, img):
        img_unshuffle = F.pixel_unshuffle(img, self.scale)
        n, c, h, w = img_unshuffle.shape
        ch = c // (self.scale ** 2)

        img_unshuffle = img_unshuffle.view(n, ch, self.scale ** 2, h, w)

        select_idx = torch.randint(0, self.scale ** 2, (n, 1, 1, h, w), device=img.device)
        select_idx = select_idx.expand(-1, ch, -1, -1, -1)

        subsampled_img = torch.gather(img_unshuffle, 2, select_idx)
        subsampled_img = subsampled_img.squeeze(2)

        mask = torch.ones_like(img_unshuffle, device=img.device)
        mask.scatter_(2, select_idx, 0)
        mask = mask.view(n, c, h, w)
        mask = F.pixel_shuffle(mask, upscale_factor=self.scale)

        return subsampled_img, mask

def SSIM(img, ground_truth):
    N = img.shape[0]
    res = []
    for k in range(N):
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(structural_similarity(img[k], ground_truth[k], data_range=data_range))
    res = np.array(res)
    return res

def PSNR(img, ground_truth):
    N = img.shape[0]
    res = []
    for k in range(N):
        mse = np.mean((img[k] - ground_truth[k]) ** 2)
        if mse == 0.:
            return float('inf')
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(20 * np.log10(data_range) - 10 * np.log10(mse))
    return np.array(res)

def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('make a new folder_'+ path)
    else:
        print(path + ' already exsits')
