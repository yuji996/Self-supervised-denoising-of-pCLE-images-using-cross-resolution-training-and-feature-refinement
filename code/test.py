import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from model import CRTFR
from utils import PSNR, SSIM
import data
import torch.nn.functional as F

def test(model_path, noisy_dir, gt_dir, output_dir, gpu=0):
    from glob import glob
    from natsort import natsorted

    DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    net = CRTFR(in_c=1, out_c=1, scale_factor=2, feature_dim=64).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    noisy_paths = natsorted(glob(os.path.join(noisy_dir, '*')))
    gt_paths = natsorted(glob(os.path.join(gt_dir, '*')))

    assert len(noisy_paths) == len(gt_paths), "number of noisy and GT images mismatch"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for i, (n_path, g_path) in enumerate(zip(noisy_paths, gt_paths)):
            noisy_img = transform(Image.open(n_path).convert('L')).unsqueeze(0).to(DEVICE)
            gt_img = transform(Image.open(g_path).convert('L')).unsqueeze(0).to(DEVICE)

            denoised_sr = net(noisy_img.float())
            denoised = F.interpolate(denoised_sr, scale_factor=0.5, mode='bilinear', align_corners=True)
            denoised = denoised.clamp(0, 1)

            if denoised.shape != gt_img.shape:
                denoised = F.interpolate(denoised, size=gt_img.shape[2:], mode='bilinear', align_corners=True)

            d_np = denoised.squeeze(0).cpu().numpy()
            g_np = gt_img.squeeze(0).cpu().numpy()

            psnr = PSNR(d_np[np.newaxis, ...], g_np[np.newaxis, ...])
            psnr_values.append(psnr[0])

            ssim = SSIM(d_np[np.newaxis, ...], g_np[np.newaxis, ...])
            ssim_values.append(ssim[0])

            print(f'Image {i}: PSNR={psnr[0]:.2f}, SSIM={ssim_values[-1]:.4f}')

            out_img = d_np
            out_img = (out_img * 255).astype(np.uint8)
            Image.fromarray(out_img, mode='L').save(os.path.join(output_dir, f'{i+21}.png'))

    print(f'\nAverage PSNR: {np.mean(psnr_values):.2f} dB')
    print(f'Average SSIM: {np.mean(ssim_values):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='Directory containing noisy test images')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth test images')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save denoised images')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')

    args = parser.parse_args()

    test(args.model_path, args.noisy_dir, args.gt_dir, args.output_dir, args.gpu)
