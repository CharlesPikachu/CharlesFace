# Author:
# 	Charles
# Function:
# 	Provide triplets for training.
import os
import random
import numpy as np
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset


# Without labels.
class TripletImageDataset(Dataset):
	def __init__(self, root, transform=None, max_pairs=1000):
		self.root = root
		self.transform = transform
		self.max_pairs = max_pairs
		self.names = sorted(os.listdir(root))
		self.images = [img for name in self.names for img in os.listdir(os.path.join(root, name))]
	def create_triplet(self):
		while True:
			class_1, class_2 = random.sample(self.names, 2)
			if len(os.listdir(os.path.join(self.root, class_1))) > 1:
				if class_1 != class_2:
					break
		class1Imgs = os.listdir(os.path.join(self.root, class_1))
		class2Imgs = os.listdir(os.path.join(self.root, class_2))
		while True:
			image_a, image_p = random.sample(class1Imgs, 2)
			if image_a != image_p:
				break
		image_n = random.sample(class2Imgs, 1)[0]
		image_a, image_p, image_n = os.path.join(self.root, class_1, image_a), os.path.join(self.root, class_1, image_p), os.path.join(self.root, class_2, image_n)
		return image_a, image_p, image_n
	def __getitem__(self, idx):
		# avoid bad images.
		while True:
			try:
				anchor_path, positive_path, negative_path = self.create_triplet()
				anchor = Image.open(anchor_path).convert('RGB')
				positive = Image.open(positive_path).convert('RGB')
				negative = Image.open(negative_path).convert('RGB')
				break
			except:
				continue
		if self.transform:
			anchor = self.transform(anchor)
			positive = self.transform(positive)
			negative = self.transform(negative)
		return anchor, positive, negative
	def __len__(self):
		return self.max_pairs


# With label.
class LTripletImageDataset(Dataset):
	def __init__(self, root, transform=None, max_pairs=1000):
		self.root = root
		self.transform = transform
		self.max_pairs = max_pairs
		self.names = sorted(os.listdir(root))
		self.num_classes = len(self.names)
		self.images = [img for name in self.names for img in os.listdir(os.path.join(root, name))]
	def create_triplet(self):
		while True:
			class_1, class_2 = random.sample(self.names, 2)
			class_1_index, class_2_index = self.names.index(class_1), self.names.index(class_2)
			if len(os.listdir(os.path.join(self.root, class_1))) > 1:
				if class_1 != class_2:
					break
		class1Imgs = os.listdir(os.path.join(self.root, class_1))
		class2Imgs = os.listdir(os.path.join(self.root, class_2))
		while True:
			image_a, image_p = random.sample(class1Imgs, 2)
			if image_a != image_p:
				break
		image_n = random.sample(class2Imgs, 1)[0]
		image_a, image_p, image_n = os.path.join(self.root, class_1, image_a), os.path.join(self.root, class_1, image_p), os.path.join(self.root, class_2, image_n)
		return image_a, image_p, image_n, class_1_index, class_2_index
	def __getitem__(self, idx):
		# avoid bad images.
		while True:
			try:
				anchor_path, positive_path, negative_path, class_1, class_2 = self.create_triplet()
				anchor = Image.open(anchor_path).convert('RGB')
				positive = Image.open(positive_path).convert('RGB')
				negative = Image.open(negative_path).convert('RGB')
				break
			except:
				continue
		if self.transform:
			anchor = self.transform(anchor)
			positive = self.transform(positive)
			negative = self.transform(negative)
		return anchor, positive, negative, class_1, class_2
	def __len__(self):
		return self.max_pairs


# For hard triplets select.
class STripletImageDataset(Dataset):
	def __init__(self, root, transform=None, max_pairs=1000):
		self.root = root
		self.transform = transform
		self.max_pairs = max_pairs
		self.names = sorted(os.listdir(root))
		self.num_classes = len(self.names)
		self.images = [img for name in self.names for img in os.listdir(os.path.join(root, name))]
	def create_triplet(self):
		while True:
			class_1, class_2 = random.sample(self.names, 2)
			class_1_index, class_2_index = self.names.index(class_1), self.names.index(class_2)
			if len(os.listdir(os.path.join(self.root, class_1))) > 1:
				if class_1 != class_2:
					break
		class1Imgs = os.listdir(os.path.join(self.root, class_1))
		class2Imgs = os.listdir(os.path.join(self.root, class_2))
		while True:
			image_a, image_p = random.sample(class1Imgs, 2)
			if image_a != image_p:
				break
		image_n = random.sample(class2Imgs, 1)[0]
		image_a, image_p, image_n = os.path.join(self.root, class_1, image_a), os.path.join(self.root, class_1, image_p), os.path.join(self.root, class_2, image_n)
		return image_a, image_p, image_n, class_1_index, class_2_index
	def __getitem__(self, idx):
		# avoid bad images.
		while True:
			try:
				anchor_path, positive_path, negative_path, class_1, class_2 = self.create_triplet()
				anchor = Image.open(anchor_path).convert('RGB')
				positive = Image.open(positive_path).convert('RGB')
				negative = Image.open(negative_path).convert('RGB')
				break
			except:
				continue
		if self.transform:
			anchor = self.transform(anchor)
			positive = self.transform(positive)
			negative = self.transform(negative)
		return anchor, positive, negative, class_1, class_2, anchor_path, positive_path, negative_path
	def __len__(self):
		return self.max_pairs


# With label.
class TripletFaceDataset(datasets.ImageFolder):
	def __init__(self, dir_, n_triplets, transform=None, *arg, **kwargs):
		super(TripletFaceDataset, self).__init__(dir_, transform)
		self.n_triplets = n_triplets
		print('[INFO]: Generating {} triplets...'.format(self.n_triplets))
		self.training_triplets = self.generate_triplets(self.imgs, self.n_triplets, len(self.classes))
	@staticmethod
	def generate_triplets(imgs, num_triplets, num_classes):
		def create_inds(images):
			inds = dict()
			for idx, (img_path, label) in enumerate(images):
				if label not in inds:
					inds[label] = []
				inds[label].append(img_path)
			return inds
		triplets = []
		indices = create_inds(imgs)
		for x in range(num_triplets):
			c1 = np.random.randint(0, num_classes)
			c2 = np.random.randint(0, num_classes)
			while len(indices[c1]) < 2:
				c1 = np.random.randint(0, num_classes)
			while c1 == c2:
				c2 = np.random.randint(0, num_classes)
			if len(indices[c1]) == 2:
				n1, n2 = 0, 1
			else:
				n1 = np.random.randint(0, len(indices[c1]))
				n2 = np.random.randint(0, len(indices[c1]))
				while n1 == n2:
					n2 = np.random.randint(0, len(indices[c1]))
			if len(indices[c2]) == 1:
				n3 = 0
			else:
				n3 = np.random.randint(0, len(indices[c2]))
			triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3], c1, c2])
		return triplets
	def __getitem__(self, index):
		def transform(img_path):
			img = self.loader(img_path)
			return self.transform(img)
		anchor, postive, negative, c1, c2 = self.training_triplets[index]
		img_a, img_p, img_n = transform(anchor), transform(postive), transform(negative)
		return img_a, img_p, img_n, c1, c2
	def __len__(self):
		return len(self.training_triplets)