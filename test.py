import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2


def dehaze_image(image_path):

	# data_hazy = Image.open(image_path)
	data_hazy = Image.open(image_path).convert('RGB')
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.cuda().unsqueeze(0)

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load('./snapshots/final_model.pth'))  # mysnapshots/dehazercolor.pth 4-26  5-31 City or NYU

	clean_image = dehaze_net(data_hazy)
#	torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("/")[-1])
#	torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "mytestoutput/11-6/" + image_path.split("\\")[-1])
	torchvision.utils.save_image(clean_image, "./2/" + image_path.split("\\")[-1])  # mytestoutput/11-26in/   \\to/  4-26 5-31 Hybird

if __name__ == '__main__':

	test_list = glob.glob("./1/*")  # 11-6  11-26 mytestinput/11-26in/*  5-31../testdata/201City/dusty/*
	# print(test_list)  # 4-26

	for image in test_list:
		print(image)
		star = time.time()
		dehaze_image(image)
		end = time.time()
		print("runingtime: %s seconds" % (end - star))
		print(image, "done!")

