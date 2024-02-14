import torch
import torch.nn as nn
import math


class dehaze_net(nn.Module):
    def __init__(self, resolution_multiplier=0.5, width_multiplier=0.5):
        super(dehaze_net, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        resolution_multiplier = max(0.1, min(1.0, resolution_multiplier))
        width_multiplier = max(0.1, min(1.0, width_multiplier))

        channels = int(3 * width_multiplier)
        self.resolution_multiplier = resolution_multiplier

        self.e_conv1 = nn.Conv2d(3, channels, 1, 1, 0, groups=channels, bias=True)

        state_dict = torch.load('snapshots/AODNet.pth')  # 加载原始模型权重
        self.e_conv1.weight.data = state_dict['e_conv1.weight'][:channels]
        self.e_conv1.bias.data = state_dict['e_conv1.bias'][:channels]

        self.e_conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        clean_image = self.relu((x2 * x) - x2 + 1)

        return clean_image


# class dehaze_net(nn.Module):  # 原本AODNet
#
#     def __init__(self):
#         super(dehaze_net, self).__init__()
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
#         self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
#         self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
#         self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
#         self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)
#
#     def forward(self, x):
#         source = []
#         source.append(x)
#
#         x1 = self.relu(self.e_conv1(x))
#         x2 = self.relu(self.e_conv2(x1))
#
#         concat1 = torch.cat((x1, x2), 1)
#         x3 = self.relu(self.e_conv3(concat1))
#
#         concat2 = torch.cat((x2, x3), 1)
#         x4 = self.relu(self.e_conv4(concat2))
#
#         concat3 = torch.cat((x1, x2, x3, x4), 1)
#         x5 = self.relu(self.e_conv5(concat3))
#
#         clean_image = self.relu((x5 * x) - x5 + 1)
#
#         return clean_image


class dehaze_net1(nn.Module):  # 原本AODNet

    def __init__(self):
        super(dehaze_net1, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(3, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(3, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)  # [1, 3, 512, 256]

        x1 = self.relu(self.e_conv1(x))  # ([1, 3, 512, 256])
        x2 = self.relu(self.e_conv2(x1))  # ([1, 3, 512, 256])

        #	concat1 = torch.cat((x1, x2), 1) 			# ([1, 6, 512, 256])
        x3 = self.relu(self.e_conv3(x2))  # ([1, 3, 512, 256])

        # concat2 = torch.cat((x2, x3), 1)        	# ([1, 6, 512, 256])
        x4 = self.relu(self.e_conv4(x3))  # ([1, 3, 512, 256])

        concat3 = torch.cat((x1, x2, x3, x4), 1)  # ([1, 12, 512, 256])
        x5 = self.relu(self.e_conv5(concat3))  # ([1, 3, 512, 256])

        clean_image = self.relu((x5 * x) - x5 + 1)  # ([1, 3, 512, 256])

        return clean_image


class dehaze_net2(nn.Module):  # 原本AODNet

    def __init__(self):
        super(dehaze_net2, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x3, x4), 1)
        x5 = self.relu(self.e_conv4(concat3))

        concat4 = torch.cat((x4, x5), 1)
        x6 = self.relu(self.e_conv4(concat4))

        concat5 = torch.cat((x5, x6), 1)
        x7 = self.relu(self.e_conv4(concat5))

        concat6 = torch.cat((x6, x7), 1)
        x8 = self.relu(self.e_conv4(concat6))

        concat7 = torch.cat((x1, x2, x3, x4), 1)
        x9 = self.relu(self.e_conv5(concat7))

        concat8 = torch.cat((x5, x6, x7, x8), 1)
        x10 = self.relu(self.e_conv5(concat8))

        concat9 = torch.cat((x9, x10), 1)

        x11 = self.relu(self.e_conv4(concat9))

        clean_image = self.relu((x11 * x) - x11 + 1)

        return clean_image
