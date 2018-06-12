# Author:
# 	Charlse
# Function:
# 	ResNet(in torchvision):
# 		resnet18, resnet34, resnet50, resnet101, resnet152.
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


# Parameters Explain:
# 	is_fc: 
# 		True: use fully connected layer to get feature vector.
# 		False: don't use fully connected layer to get feature vector.
# 	is_AvgPool:
# 		True: use AvgPool.
# 		False: without AvgPool.
# 	resnet:
# 		choose ResNet structure.
# 	embeddings_num:
# 		feature vector(128 or 256 is suggested.)
# 	num_classes:
# 		not None: TripletLoss + ClassifierLoss.
# 		None: TripletLoss.
# 	img_size:
# 		the size of input image.
# This is a complex code express, however, I think its extensibility is better.
class ResNet(nn.Module):
	def __init__(self, num_classes=None, embeddings_num=128, resnet='resnet34', pretrained=False, is_fc=True, img_size=224, is_AvgPool=True, is_softmax=False):
		super(ResNet, self).__init__()
		if num_classes:
			assert isinstance(num_classes, int)
		assert isinstance(embeddings_num, int)
		assert isinstance(img_size, int)
		self.is_fc = is_fc
		self.embeddings_num = embeddings_num
		self.num_classes = num_classes
		self.img_size = img_size
		self.is_AvgPool = is_AvgPool
		self.pretrained = pretrained
		self.resnet = resnet
		self.is_softmax = is_softmax
		self.centers = torch.zeros(num_classes, embeddings_num).type(torch.FloatTensor)
		# strides of all models is 32.
		self.stride = 32
		if is_fc and is_AvgPool:
			if resnet == 'resnet18':
				self.model = models.resnet18(pretrained=pretrained)
			elif resnet == 'resnet34':
				self.model = models.resnet34(pretrained=pretrained)
			elif resnet == 'resnet50':
				self.model = models.resnet50(pretrained=pretrained)
			elif resnet == 'resnet101':
				self.model = models.resnet101(pretrained=pretrained)
			elif resnet == 'resnet152':
				self.model = models.resnet152(pretrained=pretrained)
			else:
				print('[Error]:<ResNet in ResNets.py> ResNet structure unsupported...')
				exit(-1)
			kernel_size = math.ceil(img_size/self.stride)
			self.model.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
			if num_classes:
				self.model.classifier = nn.Linear(embeddings_num, num_classes)
		elif is_fc and (not is_AvgPool):
			self.model = nn.ModuleList()
			if resnet == 'resnet18':
				temp_model = models.resnet18(pretrained=pretrained)
				temp_model.fc = nn.Linear(512 * math.ceil(img_size/self.stride) * math.ceil(img_size/self.stride), embeddings_num)
			elif resnet == 'resnet34':
				temp_model = models.resnet34(pretrained=pretrained)
				temp_model.fc = nn.Linear(512 * math.ceil(img_size/self.stride) * math.ceil(img_size/self.stride), embeddings_num)
			elif resnet == 'resnet50':
				temp_model = models.resnet50(pretrained=pretrained)
				temp_model.fc = nn.Linear(2048 * math.ceil(img_size/self.stride) * math.ceil(img_size/self.stride), embeddings_num)
			elif resnet == 'resnet101':
				temp_model = models.resnet101(pretrained=pretrained)
				temp_model.fc = nn.Linear(2048 * math.ceil(img_size/self.stride) * math.ceil(img_size/self.stride), embeddings_num)
			elif resnet == 'resnet152':
				temp_model = models.resnet152(pretrained=pretrained)
				temp_model.fc = nn.Linear(2048 * math.ceil(img_size/self.stride) * math.ceil(img_size/self.stride), embeddings_num)
			else:
				print('[Error]:<ResNet in ResNets.py> ResNet structure unsupported...')
				exit(-1)
			self.model.append(temp_model.conv1)
			self.model.append(temp_model.bn1)
			self.model.append(temp_model.relu)
			self.model.append(temp_model.maxpool)
			self.model.append(temp_model.layer1)
			self.model.append(temp_model.layer2)
			self.model.append(temp_model.layer3)
			self.model.append(temp_model.layer4)
			self.model.append(temp_model.fc)
			if num_classes:
				temp_model.classifier = nn.Linear(embeddings_num, num_classes)
				self.model.append(temp_model.classifier)
		elif (not is_fc) and (not is_AvgPool):
			print('[INFO]:is_fc=False and is_AvgPool=False unsupported now...')
			exit(-1)
			self.model = nn.ModuleList()
			if resnet == 'resnet18':
				temp_model = models.resnet18(pretrained=pretrained)
				cov_stride = math.ceil(img_size/self.stride)
				temp_model.conv2 = nn.Conv2d(512, embeddings_num, kernel_size=(3, 3), stride=(cov_stride, cov_stride), padding=(1, 1), bias=False)
			elif resnet == 'resnet34':
				temp_model = models.resnet34(pretrained=pretrained)
				cov_stride = math.ceil(img_size/self.stride)
				temp_model.conv2 = nn.Conv2d(512, embeddings_num, kernel_size=(3, 3), stride=(cov_stride, cov_stride), padding=(1, 1), bias=False)
			elif resnet == 'resnet50':
				temp_model = models.resnet50(pretrained=pretrained)
				cov_stride = math.ceil(img_size/self.stride)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(cov_stride, cov_stride), padding=(1, 1), bias=False)
			elif resnet == 'resnet101':
				temp_model = models.resnet101(pretrained=pretrained)
				cov_stride = math.ceil(img_size/self.stride)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(cov_stride, cov_stride), padding=(1, 1), bias=False)
			elif resnet == 'resnet152':
				temp_model = models.resnet152(pretrained=pretrained)
				cov_stride = math.ceil(img_size/self.stride)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(cov_stride, cov_stride), padding=(1, 1), bias=False)
			else:
				print('[Error]:<ResNet in ResNets.py> ResNet structure unsupported...')
				exit(-1)
			self.model.append(temp_model.conv1)
			self.model.append(temp_model.bn1)
			self.model.append(temp_model.relu)
			self.model.append(temp_model.maxpool)
			self.model.append(temp_model.layer1)
			self.model.append(temp_model.layer2)
			self.model.append(temp_model.layer3)
			self.model.append(temp_model.layer4)
			self.model.append(temp_model.conv2)
			if num_classes:
				temp_model.classifier = nn.Linear(embeddings_num, num_classes)
				self.model.append(temp_model.classifier)
		elif (not is_fc) and is_AvgPool:
			self.model = nn.ModuleList()
			if resnet == 'resnet18':
				temp_model = models.resnet18(pretrained=pretrained)
				temp_model.conv2 = nn.Conv2d(512, embeddings_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			elif resnet == 'resnet34':
				temp_model = models.resnet34(pretrained=pretrained)
				temp_model.conv2 = nn.Conv2d(512, embeddings_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			elif resnet == 'resnet50':
				temp_model = models.resnet50(pretrained=pretrained)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			elif resnet == 'resnet101':
				temp_model = models.resnet101(pretrained=pretrained)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			elif resnet == 'resnet152':
				temp_model = models.resnet152(pretrained=pretrained)
				temp_model.conv2 = nn.Conv2d(2048, embeddings_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
			else:
				print('[Error]:<ResNet in ResNets.py> ResNet structure unsupported...')
				exit(-1)
			kernel_size = math.ceil(img_size/self.stride)
			temp_model.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
			self.model.append(temp_model.conv1)
			self.model.append(temp_model.bn1)
			self.model.append(temp_model.relu)
			self.model.append(temp_model.maxpool)
			self.model.append(temp_model.layer1)
			self.model.append(temp_model.layer2)
			self.model.append(temp_model.layer3)
			self.model.append(temp_model.layer4)
			self.model.append(temp_model.conv2)
			self.model.append(temp_model.avgpool)
			if num_classes:
				temp_model.classifier = nn.Linear(embeddings_num, num_classes)
				self.model.append(temp_model.classifier)
	def l2_norm(self, input_):
		input_size = input_.size()
		temp = torch.pow(input_, 2)
		normp = torch.sum(temp, 1).add_(1e-10)
		norm = torch.sqrt(normp)
		_output = torch.div(input_, norm.view(-1, 1).expand_as(input_))
		output = _output.view(input_size)
		return output
	def forward(self, x):
		if self.is_fc and self.is_AvgPool:
			x = self.model(x)
			self.features = self.l2_norm(x)
		elif self.is_fc and (not self.is_AvgPool):
			if self.num_classes:
				idx = 0
				for m in self.model:
					idx += 1
					if idx > len(self.model)-2:
						break
					x = m(x)
				x = x.view(x.size(0), -1)
				x = self.model[-2](x)
			else:
				idx = 0
				for m in self.model:
					idx += 1
					if idx > len(self.model)-1:
						break
					x = m(x)
				x = x.view(x.size(0), -1)
				x = self.model[-1](x)
			self.features = self.l2_norm(x)
		elif (not self.is_fc) and (not self.is_AvgPool):
			if self.num_classes:
				idx = 0
				for m in self.model:
					idx += 1
					if idx > len(self.model)-1:
						break
					x = m(x)
				x = x.view(x.size(0), -1)
			else:
				for m in self.model:
					x = m(x)
				x = x.view(x.size(0), -1)
			self.features = self.l2_norm(x)
		elif (not self.is_fc) and self.is_AvgPool:
			if self.num_classes:
				idx = 0
				for m in self.model:
					idx += 1
					if idx > len(self.model)-1:
						break
					x = m(x)
				x = x.view(x.size(0), -1)
			else:
				for m in self.model:
					x = m(x)
				x = x.view(x.size(0), -1)
			self.features = self.l2_norm(x)
		return self.features
	def forward_classifier(self, x):
		if self.num_classes:
			x = self.forward(x)
			if self.is_fc and self.is_AvgPool:
				x = self.model.classifier(x)
			elif self.is_fc and (not self.is_AvgPool):
				x = self.model[-1](x)
			elif (not self.is_fc) and (not self.is_AvgPool):
				pass
			elif (not self.is_fc) and self.is_AvgPool:
				pass
		else:
			print('[Error]:<ResNet in ResNets.py> argument (num_classes) should be assigned...')
			exit(-1)
		return x if not self.is_softmax else F.log_softmax(x, dim=1)
	def get_center_loss(self, target, alpha):
		batch_size = target.size(0)
		features_dim = self.features.size(1)
		target_expand =  target.view(batch_size, 1).expand(batch_size, features_dim)
		centers_var = Variable(self.centers)
		use_cuda = True if torch.cuda.is_available() else False
		if use_cuda:
			centers_batch = centers_var.gather(0, target_expand).cuda()
		else:
			centers_batch = centers_var.gather(0, target_expand)
		criterion = nn.MSELoss()
		center_loss = criterion(self.features, centers_batch)
		diff = centers_batch - self.features
		unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
		appear_times = torch.from_numpy(unique_count).gather(0, torch.from_numpy(unique_reverse))
		appear_times_expand = appear_times.view(-1, 1).expand(batch_size, features_dim).type(torch.FloatTensor)
		diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
		diff_cpu = alpha * diff_cpu
		for i in range(batch_size):
			self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())
		return center_loss, self.centers
	def load_weights(self, checkpoint=None, with_linear=False, is_center=False):
		if checkpoint is not None:
			if with_linear:
				if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
					self.load_state_dict(checkpoint['state_dict'])
				else:
					own_state = self.state_dict()
					for name, param in checkpoint['state_dict'].items():
						if "classifier" not in name:
							if isinstance(param, Parameter):
								param = param.data
							own_state[name].copy_(param)
				if is_center:
					self.centers = checkpoint['centers']
			else:
				own_state = self.state_dict()
				for name, param in checkpoint['state_dict'].items():
					if ("classifier" not in name) and ("fc" not in name):
						if isinstance(param, Parameter):
							param = param.data
						own_state[name].copy_(param)
				if is_center:
					self.centers = checkpoint['centers']
	def weights_init(self, m):
		pass