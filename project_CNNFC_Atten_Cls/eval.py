
# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from dataloader.huaWeiEmoLoader import huaWeiEmoDataloader
from models.CNNFC import AudioNet
import json
with open("config.json") as json_file:
	config = json.load(json_file)

class Evaluator():

	def __init__(self, phase):
		self.phase = phase
		self.criterion = nn.NLLLoss()

	def infoLogger(self, output, Y, fo):
		print('[{}]'.format(self.phase))
		fo.write('[{}]\n'.format(self.phase))
		loss = self.criterion(output, Y)
		acc = accuracy_score(torch.argmax(output, dim=1).cpu(), Y.cpu())
		print('{}/{}: loss: {:.3f}, acc: {:.3f}'.format('--', '--', loss, acc))
		fo.write('{}/{}: loss: {:.3f}, acc: {:.3f}\n'.format('--', '--', loss, acc))
		cm = confusion_matrix(torch.argmax(output, dim=1).cpu(), Y.cpu())
		cm = cm/cm.sum(axis=1)[:, np.newaxis]
		for i in range(len(config["TRAGET_INFO"])):
			print("{:12s}\t".format(config["TRAGET_INFO"][i]), end=' ')
			fo.write("{:12s}\t".format(config["TRAGET_INFO"][i]))
			for j in range(len(config["TRAGET_INFO"])):
				print("{:.3f}\t".format(cm[i][j]), end=' ')
				fo.write("{:.3f}\t".format(cm[i][j]))
			print("\n")
			fo.write("\n")
		return loss, acc

	def evaluation(self, X, Y, model, fo):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		torch_dataset = Data.TensorDataset(X, Y)
		dataLoader = Data.DataLoader(dataset = torch_dataset, batch_size = config["VALID_BATCH_SIZE"], shuffle = False)
		model =model.to(device)
		model.eval()
		with torch.no_grad():
			output = []
			for _, (batch_x, batch_y) in enumerate(dataLoader):
				batch_x = batch_x.to(device)
				output_batch = model(batch_x)
				output.append(output_batch)
			output = torch.cat(output, axis = 0)
		loss, acc = self.infoLogger(output, Y.to(device), fo)
		return loss, acc

class Tester():

	def __init__(self, expName):
		self.expName = expName
		self.phase = 'test'
		self.logPath = os.path.join(config["PROJECT_PATH"][config["ENV"]], 'logs', self.expName)
		self.fo_test = open(os.path.join(self.logPath, 'test.txt'), 'w+')
		dataload = huaWeiEmoDataloader()
		self.X_test, self.Y_test = dataload.datasetLoader('test')

	def test(self):
		lastModel = torch.load(os.path.join(self.logPath, 'lastModel.pkl'))
		bestAccModel = torch.load(os.path.join(self.logPath, 'bestModel_acc.pkl'))
		evaluator = Evaluator('lastModel')
		_, _ = evaluator.evaluation(self.X_test, self.Y_test, lastModel, self.fo_test)
		evaluator = Evaluator('bestAccModel')
		_, _ = evaluator.evaluation(self.X_test, self.Y_test, bestAccModel, self.fo_test)

	def __del__(self):
		self.fo_test.close()



if __name__ == '__main__':
	tester = Tester("0107")
	tester.test()