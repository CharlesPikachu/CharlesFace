# Author:
# 	Charles
# Function:
# 	For test LFW dataset.
import os
import numpy as np
import torchvision.datasets as datasets


class LFWDataset(datasets.ImageFolder):
	def __init__(self, dir_, pairs_path, transform=None):
		super(LFWDataset, self).__init__(dir_, transform)
		self.pairs_path = pairs_path
		self.images = self.get_lfw_paths(dir_)
	def read_lfw_pairs(self, pairs_filename):
		pairs = []
		with open(pairs_filename, 'r') as f:
			for line in f.readlines()[1:]:
				pair = line.strip().split()
				pairs.append(pair)
		return np.array(pairs)
	def get_lfw_paths(self, lfw_dir, file_ext='.png'):
		pairs = self.read_lfw_pairs(self.pairs_path)
		skipped_num = 0
		path_list = []
		issame_list = []
		for i in range(len(pairs)):
			pair = pairs[i]
			if len(pair) == 3:
				path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + ('%d' % int(pair[1])).zfill(4) + file_ext)
				path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + ('%d' % int(pair[2])).zfill(4) + file_ext)
				issame = True
			elif len(pair) == 4:
				path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + ('%d' % int(pair[1])).zfill(4) + file_ext)
				path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + ('%d' % int(pair[3])).zfill(4) + file_ext)
				issame = False
			# print(path0, path1)
			if os.path.exists(path0) and os.path.exists(path1):
				path_list.append((path0, path1, issame))
				issame_list.append(issame)
			else:
				skipped_num += 1
		if skipped_num > 0:
			print('[Warning]: %s <LFWDataset.get_lfw_paths in LFWDataset.py>images lost...' % skipped_num)
		return path_list
	def __getitem__(self, index):
		def transform(img_path):
			img = self.loader(img_path)
			# from PIL import Image
			# img = Image.open(img_path).convert('RGB')
			return self.transform(img)
		path_1, path_2, issame = self.images[index]
		img1, img2 = transform(path_1), transform(path_2)
		return img1, img2, issame
	def __len__(self):
		return len(self.images)