
import os
import sys
import json
import torch
import torch.nn as nn
from torchsummary import summary
with open(os.path.join(sys.path[0], 'config.json')) as json_file:
	config = json.load(json_file)

class AudioNet(nn.Module):

	def __init__(self):
		super(AudioNet, self).__init__() # (128, 1292)
		self.num_output = 5

		if(config["SAL0"] == "Wi"):
			self.W_i = nn.Parameter(torch.randn(40,1))                  # with channel num 16
			self.bi = nn.Parameter(torch.randn(1))
			self.SmSA0 = nn.Softmax(dim = 1)
		self.conv0 = nn.Sequential(nn.Conv1d(128, 32, kernel_size=8), nn.MaxPool1d(4, stride=4), nn.ReLU(), nn.BatchNorm1d(32),)
		if(config["CAL1"] == "FC"):
			self.CA1_avg_pool = nn.AdaptiveAvgPool1d(1)
			self.CA1 = nn.Sequential(nn.Linear(in_features=32, out_features=32), nn.Softmax(dim=1),)
		if(config["CAL1"] == "Conv"):
			self.CA1_avg_pool = nn.AdaptiveAvgPool1d(1)
			self.conv_du1 = nn.Sequential(nn.Conv1d(32, 32//2, 1, bias=True), nn.ReLU(inplace=True), nn.Conv1d(32//2, 32, 1, bias=True), nn.Sigmoid())
		self.conv1 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=8), nn.MaxPool1d(4, stride=4), nn.ReLU(), nn.BatchNorm1d(16),)
		if(config["CAL2"] == "FC"):
			self.CA2_avg_pool = nn.AdaptiveAvgPool1d(1)
			self.CA2 = nn.Sequential(nn.Linear(in_features=16, out_features=16), nn.Softmax(dim=1),)
		if(config["CAL2"] == "Conv"):
			self.CA2_avg_pool = nn.AdaptiveAvgPool1d(1)
			self.conv_du2 = nn.Sequential(nn.Conv1d(16, 16//2, 1, bias=True), nn.ReLU(inplace=True), nn.Conv1d(16//2, 16, 1, bias=True),nn.Sigmoid())
		self.fc0 = nn.Sequential(nn.Linear(in_features=1248, out_features=64), nn.Tanh(), nn.Dropout(),  nn.Linear(in_features=64, out_features=self.num_output),)
		self.logsoftmax = nn.LogSoftmax(dim=1)
		self.apply(self._init_weights)

	def forward(self, x):
		if(config["SAL0"] == "Wi"):
			z0 = x.permute(0, 2, 1)                      #(N, L, C)
			alpha0 = (self.SmSA0((torch.matmul(z0, self.W_i) + self.bi).squeeze(-1))).unsqueeze(-1)  #(N, L, 1)-->(N, L)
			x = (z0*alpha0).permute(0, 2, 1)
		x = self.conv0(x)                                                                   #(N, 32, 321)
		if(config["CAL1"] == "FC"):
			beta1 = self.CA1_avg_pool(x).squeeze(-1)     #(N, 32, 1) --> (N, 32)
			beta1 = self.CA1(beta1).unsqueeze(-1)
			x = x*beta1
		if(config["CAL1"] == "Conv"):
			beta1 = self.CA1_avg_pool(x)                  #(N, 32, 1) --> (N, 32)
			beta1 = self.conv_du1(beta1)
			x = x*beta1
		x = self.conv1(x)                                                                    #(N, 16, 78)
		if(config["CAL2"] == "FC"):
			beta2 = self.CA2_avg_pool(x).squeeze(-1)     #(N, 16, 1) --> (N, 16)
			beta2 = self.CA2(beta2).unsqueeze(-1)        #(N, 16, 1)
		x = x*beta2                                  #(N, 16, 78)
		if(config["CAL2"] == "Conv"):
			beta2 = self.CA2_avg_pool(x)                 #(N, 16, 1) 
			beta2 = self.conv_du2(beta2)
			x = x*beta2
		flatten = x.view(x.size(0), -1)
		# print(flatten.shape)
		out = self.logsoftmax(self.fc0(flatten))
		#x = F.log_softmax(x, dim=1)             # output (N, 5)
		return out

	def _init_weights(self, layer) -> None:
		if(isinstance(layer, nn.Conv1d)):
			nn.init.kaiming_uniform_(layer.weight)
		elif(isinstance(layer, nn.Linear)):
			nn.init.xavier_uniform_(layer.weight)


if __name__ == '__main__':
	net = AudioNet()
	print(net)
	net = net.cuda()
	summary(net, (128, 1292))