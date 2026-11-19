import data
import torch
import model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torchvision import transforms
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

if __name__ == '__main__':

    writer = SummaryWriter('./log')

    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str, required=True,
                        help='the file path of input train noisy data')
    parser.add_argument('-val_path', type=str, required=True,
                        help='the file path of input validation noisy data')
    parser.add_argument('-val_gt_path', type=str, required=True,
                        help='the file path of input validation ground truth data')
    parser.add_argument('-model_save_path', type=str, default='model/',
                        help='the path of saved model weight')
    parser.add_argument('-save_path', type=str, default='output/',
                        help='the path of validation output')

    parser.add_argument('-lr', type = float, default=1e-4, dest= 'lr',
                        help = 'learning rate')
    parser.add_argument('-epoch', type = int, default= 1000, dest= 'epoch',
                        help = 'the total number of training epoches')
    parser.add_argument('-summary_epoch', type = int, default=100, dest='summary_epoch',
                        help = 'the current model will be saved per summary_epoch')
    parser.add_argument('-val_epoch', type=int, default=10, dest='val_epoch',
                        help='validate on the validation set every val_epoch epochs')
    parser.add_argument('-bs', type = int, default = 16, dest='batch_size',
                        help = 'the number of training batches')
    parser.add_argument('-ps', type = int, default=256, dest= 'patch_size',
                        help = 'the patch size for training images')
    parser.add_argument('-gpu',type = int, default=0,dest='gpu',
                        help = 'the number of GPU')

    args = parser.parse_args()
    model_save_path = args.model_save_path
    save_path = args.save_path
    lr = args.lr
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    val_epoch = args.val_epoch
    batch_size = args.batch_size
    patch_size = args.patch_size

    utils.makedir(model_save_path)
    utils.makedir(save_path)

    DEVICE = torch.device('cuda:{}'.format(str(args.gpu) if torch.cuda.is_available() else 'cpu'))

    train_loader = data.loader_train(args.train_path, patch_size=patch_size, batch_size=batch_size)
    val_loader = data.loader_val(args.val_path, batch_size=10)
    val_gt_loader = data.loader_val(args.val_gt_path, batch_size=10)

    Sub_sampler = utils.Sub_sampler(2, DEVICE)

    net = model.CRTFR(in_c=1, out_c=1, scale_factor=2, feature_dim=64).to(DEVICE)

    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr = lr)

    tqdm_loop = tqdm(range(epoch), ncols = 120)

    for e in tqdm_loop:

        net.train()
        loss_train = 0

        for i, (x1) in enumerate(train_loader):

            x1 = x1.to(DEVICE)
            x1_in, mask = Sub_sampler.sample_img(x1)
            mask = mask.to(DEVICE)
            img_pred = net(x1_in.float())

            img_pred_mask = img_pred * mask
            x1_mask = x1 * mask

            loss = loss_fun(img_pred_mask.float(), x1_mask.float())
            loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tqdm_loop.set_description_str('(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.6f}'.format(e + 1,
                                                                                      epoch,
                                                                                      i + 1,
                                                                                      len(train_loader),
                                                                                      current_lr,
                                                                                      loss.item()))

        writer.add_scalar('MSE_train', loss_train / len(train_loader), e + 1)

        if (e + 1) % val_epoch == 0 or (e + 1) % summary_epoch == 0:
            net.eval()
            with torch.no_grad():
                psnr_values = []
                ssim_values = []
                pred_whole = []

                for (x1, gt) in zip(val_loader, val_gt_loader):
                    x1_in = x1.to(DEVICE)
                    gt = gt.to(DEVICE)

                    pred = net(x1_in)
                    pred = F.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=True)

                    if gt.shape[-2:] != pred.shape[-2:]:
                        gt = F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=True)

                    for i in range(pred.shape[0]):
                        pred_img = pred[i].cpu().numpy()
                        gt_img = gt[i].cpu().numpy()

                        psnr = utils.PSNR(pred_img, gt_img)
                        ssim = utils.SSIM(pred_img, gt_img)
                        psnr_values.append(psnr[0])
                        ssim_values.append(ssim[0])

                    pred_whole.append(pred.cpu().numpy())

                avg_psnr = np.mean(psnr_values)
                avg_ssim = np.mean(ssim_values)

                writer.add_scalar('PSNR', avg_psnr, e + 1)
                writer.add_scalar('SSIM', avg_ssim, e + 1)
                print(f'PSNR: {avg_psnr:.2f}')
                print(f'SSIM: {avg_ssim:.4f}')

                if (e + 1) % summary_epoch == 0:
                    pred_whole = np.concatenate(pred_whole, axis=0)
                    for idx, pred_img in enumerate(pred_whole):
                        pred_img = torch.from_numpy(pred_img)
                        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
                        pred_img = (pred_img * 255).type(torch.uint8)
                        pred_img_pil = transforms.ToPILImage()(pred_img)
                        pred_img_pil.save(os.path.join(save_path, f'CRTFR_denoised_{e + 1}_{idx}.png'))

                    torch.save(net.state_dict(), os.path.join(model_save_path, f'model_param_{e + 1}.pth'))

        writer.flush()
