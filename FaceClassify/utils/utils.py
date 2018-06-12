# Author: 
# 	Charles
# Function:
# 	Provide some utils functions
import os
import time
import torch
import collections
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Function
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


# Load/Save weights
# --------------------------------------------------------------------------------------------------------------
# Function:
# 	Save checkpoint.
def save_checkpoint(model_state, optimizer_state, filename):
	state = dict(model_state=model_state, optimizer_state=optimizer_state)
	torch.save(state, filename)
# --------------------------------------------------------------------------------------------------------------


# Image process
# --------------------------------------------------------------------------------------------------------------
# Function:
# 	PIL Image convert to Torch.
def image2torch(img):
	width = img.width
	height = img.height
	img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
	img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
	img = img.view(1, 3, height, width)
	img = img.float().div(255.0)
	return img

# Function:
# 	Rescales the input PIL.Image to given 'size'.
def scale_fun(size, img, interpolation=Image.BILINEAR):
	if isinstance(size, int):
		w, h = img.size
		if (w<=h and w==size) or (h<=w and h==size):
			return img
		elif w < h:
			return img.resize((size, int(size*h/w)), interpolation)
		else:
			return img.resize((int(size*w/h), size), interpolation)
	elif isinstance(size, collections.Iterable) and len(size)==2:
		return img.resize(size, interpolation)
	else:
		print('[Error]: <scale fun in utils.py> size type error...')
		exit(-1)
class Scale(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size)==2)
		self.size = size
		self.interpolation = interpolation
	def __call__(self, img):
		if isinstance(self.size, int):
			w, h = img.size
			if (w<=h and w==self.size) or (h<=w and h==self.size):
				return img
			elif w < h:
				return img.resize((self.size, int(self.size*h/w)), self.interpolation)
			else:
				return img.resize((int(self.size*w/h), self.size), self.interpolation)
		else:
			return img.resize(self.size, self.interpolation)
# Function:
# 	Resize the input PIL.Image to given 'size'.
class Resize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size)==2)
		self.size = size
		self.interpolation = interpolation
	def __call__(self, img):
		if isinstance(self.size, int):
			return img.resize((self.size, self.size), self.interpolation)
		else:
			return img.resize(self.size, self.interpolation)
# --------------------------------------------------------------------------------------------------------------


# Calculate L-Normalization
# --------------------------------------------------------------------------------------------------------------
# Function:
# 	diff = x1 - x2  ->  vector
# 	(diff1^p + diff2^p + ... + diffn^p) ^ 1/p
class PairwiseDistance(Function):
	def __init__(self, p):
		super(PairwiseDistance, self).__init__()
		self.norm = p
	def forward(self, x1, x2):
		assert x1.size() == x2.size()
		eps = 1e-4 / x1.size(1)
		diff = torch.abs(x1 - x2)
		out = torch.pow(diff, self.norm).sum(dim=1)
		return torch.pow(out+eps, 1./self.norm)

# Function:
# 	Store some values class.
class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
# --------------------------------------------------------------------------------------------------------------


# Eval while training
# TP: True Positive.
# TN: True Negative.
# FP: False Positive.
# FN: False Negative.
# TPR: True Positive Rate. -> TP / (TP+FN)
# TNR: True Negative Rate. -> TN / (TN+FP)
# FPR: False Positive Rate. -> FP / (FP+TN)
# FNR: False Negative Rate. -> FN / (TP+FN)
# --------------------------------------------------------------------------------------------------------------
# Function:
# 	calculate accuracy.
def calculate_accuracy(threshold, distances, issame):
	pred_issame = np.less(distances, threshold)
	TP = np.sum(np.logical_and(pred_issame, issame))
	FP = np.sum(np.logical_and(pred_issame, np.logical_not(issame)))
	TN = np.sum(np.logical_and(np.logical_not(pred_issame), np.logical_not(issame)))
	FN = np.sum(np.logical_and(np.logical_not(pred_issame), issame))
	TPR = 0 if (TP+FN == 0) else float(TP) / float(TP+FN)
	FPR = 0 if (FP+TN == 0) else float(FP) / float(FP+TN)
	acc = float(TP+TN) / distances.size
	return TPR, FPR, acc

# Function:
# 	calculate roc.
def calculate_roc(thresholds, distances, labels, K=10):
	assert len(labels) == len(distances)
	num_pairs = len(labels)
	num_thresholds = len(thresholds)
	k_fold = KFold(n_splits=K, shuffle=True)
	TPRs = np.zeros((K, num_thresholds))
	FPRs = np.zeros((K, num_thresholds))
	accuracy = np.zeros((K))
	indices = np.arange(num_pairs)
	for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
		acc_train = np.zeros((num_thresholds))
		# Get the best threshold for the fold.
		for threshold_idx, threshold in enumerate(thresholds):
			tmp1, tmp2, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
		best_threshold_index = np.argmax(acc_train)
		for threshold_idx, threshold in enumerate(thresholds):
			TPRs[fold_idx, threshold_idx], FPRs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
		tmp1, tmp2, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])
		TPR = np.mean(TPRs, 0)
		FPR = np.mean(FPRs, 0)
	return TPR, FPR, accuracy

# Function:
# 	Called to evaluate trained model.
def evaluate(distances, labels, K=10):
	thresholds = np.arange(0, 30, 0.01)
	TPR, FPR, accuracy = calculate_roc(thresholds, distances, labels, K=K)
	return TPR, FPR, accuracy
# --------------------------------------------------------------------------------------------------------------


# Other helper
# --------------------------------------------------------------------------------------------------------------
# Function:
# 	GPU -> CPU.
# 	--Float.
def convert2cpu(gpu_matrix):
	return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
# 	--Long.
def convert2cpu_long(gpu_matrix):
	return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

# Function:
# 	Output time now + message.
def logging(message):
	print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

# Function:
# 	Plot ROC.
def plot_roc(FPR, TPR, savefile, figure_name='roc.png'):
	roc_auc = auc(FPR, TPR)
	fig = plt.figure()
	plt.plot(FPR, TPR, color='darkorange', lw=2, label='ROC curve (area=%0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc='lower right')
	fig.savefig(os.path.join(savefile, figure_name), dpi=fig.dpi)

# Function:
# 	Calculate the number of parameters.
def summary(net):
	num_params = 0
	for name, param in net.named_parameters():
		num_params += param.numel()
	print('[INFO]: Total number of parameters: %d' % num_params)

# Function:
# 	Save/Read triplets generated before training.
def save_triplets(triplets, save_path):
	f = open(save_path, 'w')
	for triplet in triplets:
		for t in triplet:
			f.write(str(t) + ' ')
		f.write('\n')
	f.close()
def read_triplets(save_path):
	f = open(save_path, 'r')
	triplets = []
	all_lines = f.readlines()
	for all_line in all_lines:
		all_line = all_line.strip()
		temp = all_line.split(' ')
		assert len(temp) == 5
		triplets.append([temp[0], temp[1], temp[2], int(temp[3]), int(temp[4])])
	f.close()
	return triplets
# --------------------------------------------------------------------------------------------------------------