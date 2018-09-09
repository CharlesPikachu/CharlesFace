# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.standard_utils import read_truths_args, read_truths
from utils.image_preprocess import *


class listDataset(Dataset):
	def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
		with open(root, 'r') as file:
			self.lines = file.readlines()
		if shuffle:
			random.shuffle(self.lines)
		self.nSamples  = len(self.lines)
		self.transform = transform
		self.target_transform = target_transform
		self.train = train
		self.shape = shape
		self.seen = seen
		self.batch_size = batch_size
		self.num_workers = num_workers
	def __len__(self):
		return self.nSamples
	def __getitem__(self, index):
		# index range error
		assert index <= len(self)
		imgpath = self.lines[index].rstrip()
		# for multiscale training.
		if self.train and index % 64 == 0:
			if self.seen < 4000*64:
				width = 13 * 32
				self.shape = (width, width)
			elif self.seen < 8000*64:
				width = (random.randint(0, 3) + 13) * 32
				self.shape = (width, width)
			elif self.seen < 12000*64:
				width = (random.randint(0, 5) + 12) * 32
				self.shape = (width, width)
			elif self.seen < 16000*64:
				width = (random.randint(0, 7) + 11) * 32
				self.shape = (width, width)
			else:
				width = (random.randint(0, 9) + 10) * 32
				self.shape = (width, width)
		if self.train:
			jitter = 0.2
			# H of HSV
			hue = 0.1
			# S of HSV
			saturation = 1.5
			# V of HSV
			exposure = 1.5
			img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
		else:
			img = Image.open(imgpath).convert('RGB')
			if self.shape:
				img = img.resize(self.shape)
			labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
			# the num of boxes max 50.
			label = torch.zeros(50*5)
			# 8 pixels for smallest face.
			try:
				tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
			except Exception:
				print('[Warning]:%s has no data...' % labpath)
				tmp = torch.zeros(1, 5)
			tmp = tmp.view(-1)
			# tmp size
			tsz = tmp.numel()
			if tsz > 50*5:
				label = tmp[0: 50*5]
			elif tsz > 0:
				label[0: tsz] = tmp
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)
		if self.num_workers > 0:
			self.seen = self.seen + self.num_workers
		elif self.num_workers == 0:
			self.seen += 1
		else:
			print('[Error]:num_workers must greater than zero...')
		return (img, label)



if __name__ == '__main__':
	batch_size = 1
	testlist = '2007_test.txt'
	init_width = 416
	init_height = 416
	from torchvision import transforms
	from nets.darknet import Darknet
	cfgfile = './cfg/yolov3.cfg'
	num_workers = 4
	model = Darknet(cfgfile)
	cur_model = model
	kwargs = {'num_workers': num_workers, 'pin_memory': True}
	listDataset(testlist, 
				shape=(init_width, init_height),
				shuffle=False,
				transform=transforms.Compose([transforms.ToTensor(),]),
				train=False)[0]