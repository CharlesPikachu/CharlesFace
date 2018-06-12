# Author:
# 	Charles
# Function:
# 	Provide triplets for training.
import torchvision.datasets as datasets
import numpy as np


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