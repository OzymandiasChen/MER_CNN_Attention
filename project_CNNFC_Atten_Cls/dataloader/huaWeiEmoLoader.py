
import os
import sys
import json
import librosa
import numpy as np
from random import shuffle
import torch
import xlrd
import pandas as pd
with open(os.path.join(sys.path[0], 'config.json')) as json_file:
	config = json.load(json_file)
# with open('config.json') as json_file:
# 	config = json.load(json_file)

'''
# classNameList = ['Angry', 'Empowering', 'Somber', 'Stirring', 'Upbeat', 'Peaceful']
|-- train/
|-- valid/
|-- test/
'''	
class huaWeiEmoDataloader():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): init function for path setters.
			(2). audio2Mel(self, datasetName): audio2mel for a certain dataset
			(3). datasetLoader(self, datasetName): load a certain dataset for 'train'/'valid'/'test'.
		Description:
			(a). all functions could be used separately.
		Using:
			(a). If it is the first time to use, call 'huaWeiEmoDataloader().audio2Mel()' for loading and saving 10s mel.
			(b). call 'huaWeiEmoDataloader().datasetLoader(datasetName)' to load a certain dataset.
	'''
	def __init__(self):
		'''
		init function for path setters.
		'''
		self.datasetPath = config["HUAWEIEMO_PATH"][config["ENV"]]
		self.melPklPath = os.path.join(self.datasetPath, config["AUDIO_PROCESSING_METHOD"])
		if not os.path.exists(self.melPklPath):
			os.makedirs(self.melPklPath)

	def audio2Mel(self, datasetName):
		'''
		audio2mel for a certain dataset
		Input:
			datasetName: 'train'/'valid'/'test'
		'''
		audioFolderPath = os.path.join(self.datasetPath, datasetName.lower())
		melFolderPath  =  os.path.join(self.melPklPath, datasetName.lower())
		if not os.path.exists(melFolderPath):
			os.makedirs(melFolderPath)
		for audioFile in os.listdir(audioFolderPath):
			audioFilePath = os.path.join(audioFolderPath, audioFile)
			y, _ = librosa.load(audioFilePath, sr = config["SR"])
			melSpectro = librosa.feature.melspectrogram(y, sr = config["SR"])
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, os.path.join(melFolderPath, audioFile.split('.')[0]+'.pkl'))
			print(audioFile, ':  ', logMelSpectro.shape)

	def datasetLoader(self, datasetName):
		'''
		load a certain dataset for 'train'/'valid'/'test'.
		'''
		melFolderPath  =  os.path.join(self.melPklPath, datasetName.lower())
		X = []
		Y = []
		for pklFile in os.listdir(melFolderPath):
			logMelSpectro_audio = torch.load(os.path.join(melFolderPath, pklFile))
			tag = int(pklFile.split('.')[0][-1])
			X.append(logMelSpectro_audio)
			Y.append(tag)
		X = torch.from_numpy(np.array(X))
		Y = torch.from_numpy(np.array(Y))
		if(config["MODE"] == "2D"):
			X = X.unsqueeze(1)
		X = X.float()
		Y = Y.long()
		print('-------------Loading {}--------------\n'.format(datasetName), X.shape, '	', Y.shape)
		return X, Y


if __name__ == '__main__':
	dataloader = huaWeiEmoDataloader()
	dataloader.datasetLoader('train')
	# dataloader.datasetLoader('valid')
	# dataloader.datasetLoader('test')

