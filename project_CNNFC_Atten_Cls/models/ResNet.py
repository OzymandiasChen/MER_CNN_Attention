
import os
import sys
import json
import torch
# with open(os.path.join(sys.path[0], 'config.json')) as json_file:
# 	config = json.load(json_file)
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(ResidualBlock, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel)
		)
		self.shortcut = nn.Sequential()
		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(outchannel)
			)

	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, ResidualBlock, num_classes=6):
		super(ResNet, self).__init__()

		self.inchannel = 16
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
		)
		self.layer1 = self.make_layer(ResidualBlock, 16,  2, stride=1)
		self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
		self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
		self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(128, num_classes)
		self.logsoftmax = nn.LogSoftmax(dim=1)


	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.gap(out)
		#out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out = self.logsoftmax(out)
		return out


def ResNet18():

    return ResNet(ResidualBlock)

if __name__ == '__main__':
	net = ResNet18()
	print(net)
	net = net.cuda()
	# summary(net, (1, 128, 1292))
	summary(net, (1, 224, 224))