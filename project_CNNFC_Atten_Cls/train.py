
# coding: utf-8
import os
import json
import numpy as np
from models.CNNFC import AudioNet
from models.ResNet import ResNet18
from dataloader.huaWeiEmoLoader import huaWeiEmoDataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from eval import Evaluator
from sklearn.metrics import accuracy_score
# from utils import LossLogger
with open("config.json") as json_file:
	config = json.load(json_file)


class Trainer():

	def __init__(self, expName):
		if(config["MODEL"] == 'CNNFC'):
			self.model = AudioNet()
		elif(config["MODEL"] == 'ResNet'):
			self.model = ResNet18()
		self.optimizer = optim.Adam(self.model.parameters(), lr = config["LR"])
		self.criterion = nn.NLLLoss()
		self.epochNum = config["EPOCH_NUM"]
		self.bestLoss = float("Inf")
		self.bestAcc = 0
		self.expName = expName
		self.logPath = os.path.join(config["PROJECT_PATH"][config["ENV"]], 'logs', self.expName)
		if not os.path.exists(self.logPath):
			os.makedirs(self.logPath)
		self.writter = SummaryWriter(log_dir = os.path.join(self.logPath, 'tensorboard'))
		self.nonbetterCount = 0
		self.patience = config["EARLY_STOPPING_PATIENCE"]

		dataload = huaWeiEmoDataloader()
		X_train, Y_train = dataload.datasetLoader('train')
		train_torch_dataset = Data.TensorDataset(X_train, Y_train)
		self.train_loader = Data.DataLoader(dataset = train_torch_dataset, batch_size = config["TRAIN_BATCH_SIZE"], shuffle = True)
		self.X_valid, self.Y_valid = dataload.datasetLoader('valid')
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = self.model.to(self.device)
		self.fo = open(os.path.join(self.logPath, 'trainLog.txt'), 'w+')
		self.targetInfo = config["TRAGET_INFO"]
		self.writter = SummaryWriter(log_dir = os.path.join(self.logPath, 'tensorboard'))

	def betterSaver(self, acc_valid):
		if(acc_valid >= self.bestAcc):
			self.bestAcc = acc_valid
			torch.save(self.model.cpu(), os.path.join(self.logPath, 'bestModel_acc.pkl'))
			print('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
			self.fo.write('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
			self.nonbetterCount = 0
		else:
			self.nonbetterCount = self.nonbetterCount + 1
		if(self.nonbetterCount == self.patience):
			print('[EARLY STOPPING!!]\n')
			self.fo.write('[EARLY STOPPING!!]\n')
			return True
		return False # continue flag

	def lossAccWritter(self, loss, acc, stepIndex, epochIndex, phase):
		'''
		Writter for loss and acc in fo and on screen.
		Input: 
			fo, loss, acc, stepIndex, epochIndex: info
			phase: writter pahse info, 'batch'/'train'
		Output:
			----
		'''
		if(phase.lower() == 'train'):
			print('[Train] ')
			self.fo.write('[Train] ')
		print('{}/{}: loss: {:.3f}, acc: {:.3f}'.format(stepIndex, epochIndex, loss, acc))
		self.fo.write('{}/{}: loss: {:.3f}, acc: {:.3f}\n'.format(stepIndex, epochIndex, loss, acc))

	def one_pass_train(self, epochIndex):
		epoch_loss = 0
		epoch_acc = 0
		self.model.train()
		self.model = self.model.to(self.device)
		for step, (batch_x, batch_y) in enumerate(self.train_loader):
			self.model.zero_grad()
			batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
			output_batch = self.model(batch_x)
			# 		X = torch.cat(X, axis=0)
			# torch.cat((x, x, x), 1)
			loss = self.criterion(output_batch, batch_y)
			loss.backward()
			self.optimizer.step()
			acc = accuracy_score(torch.argmax(output_batch, dim=1).cpu(), batch_y.cpu())
			epoch_loss += loss.item() * batch_x.shape[0]
			epoch_acc += acc * batch_x.shape[0]
			if(step % (len(self.train_loader) // 6) == 0):
				self.lossAccWritter(loss.item(), acc, step, epochIndex, 'batch')
				self.writter.add_scalar('Loss/batch', loss.item(), epochIndex*len(self.train_loader)+step)
				self.writter.add_scalar('Acc/batch', acc, epochIndex*len(self.train_loader)+step)
		torch.save(self.model.cpu(), os.path.join(self.logPath, 'lastModel.pkl'))
		return epoch_loss / len(self.train_loader.dataset), epoch_acc / len(self.train_loader.dataset)

	def train(self):
		valid_evaluator = Evaluator('valid')
		for epochIndex in range(self.epochNum):
			loss_train, acc_train = self.one_pass_train(epochIndex)
			self.lossAccWritter(loss_train, acc_train, '--', epochIndex, 'train')
			loss_valid, acc_valid = valid_evaluator.evaluation(self.X_valid, self.Y_valid, self.model, self.fo)
			self.writter.add_scalars('Loss/epoch', {'train': loss_train, 
													'valid': loss_valid.item()}, epochIndex)
			self.writter.add_scalars('Acc/epoch', {'train': acc_train, 
													'valid': acc_valid}, epochIndex)
			self.writter.flush()
			if(self.betterSaver(acc_valid) == True):
				break

	def __del__(self):
		self.writter.close()
		self.fo.close()


if __name__ == '__main__':
	trainer = Trainer("0107_FC_FC")
	trainer.train()