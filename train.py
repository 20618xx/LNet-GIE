import gc

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

from matplotlib import pyplot as plt

import dataloader
import net
import numpy as np
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_CACHE_PATH'] = '~/.cudacache'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                               config.hazy_images_path)
    print(train_dataset[0][0].shape)  # Assuming train_dataset returns a tuple (input, target)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()
    losses = []
    for epoch in range(config.num_epochs):
        print("epoch", epoch)

        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

        # if ((iteration+1) % config.display_iter) == 0:
        #     print("Loss at iteration", iteration+1, ":", loss.item())
        # if ((epoch+1) % config.snapshot_iter) == 0:
        #     torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)
            torchvision.utils.save_image(clean_image, config.sample_output_folder + str(iter_val + 1) + ".jpg")

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg")

        losses.append(loss.item())
        print(loss.item())
        torch.cuda.empty_cache()
    
    # 保存最后一个模型
    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "final_model.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str,
                        default="./data/data/clear/")  # Cityscapes or NYU   ../traindata/1764NYU/label/
    parser.add_argument('--hazy_images_path', type=str, default="./data/data/hazy/")  # data/Cityscapes/dusty/
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30)  # 10  100 4-26
    parser.add_argument('--train_batch_size', type=int, default=8)  # 8
    parser.add_argument('--val_batch_size', type=int, default=1)  # 8
    parser.add_argument('--num_workers', type=int, default=0)  # 4
    parser.add_argument('--display_iter', type=int, default=1)  # 10  50fornyu, 20forcity 4-26
    parser.add_argument('--snapshot_iter', type=int, default=1)  # 200
    parser.add_argument('--snapshots_folder', type=str,
                        default="./snapshots/")  # snapthots  # mysnapshots/Cityscapes/ 4-26
    parser.add_argument('--sample_output_folder', type=str, default="./samples/")  # mysamples

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)


