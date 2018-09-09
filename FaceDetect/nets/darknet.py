# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
from utils.cfg_utils import *
from utils.standard_utils import *
from loss_module.region_loss import RegionLoss
from layers.yolo_layer import YoloLayer
from cfg import *


# pad:
# 	pre-Tensor <Right and Bottom> expand 1 dim.
# 	(the same as pre-final <Right and Bottom>)
# maxpool2d:
# 	stride: 1
# 	k_size: (2, 2)
# Input:
# 	(batch_size, in_channels, w, h)
# Output:
# 	the same as Input.
class MaxPoolStride1(nn.Module):
	def __init__(self):
		super(MaxPoolStride1, self).__init__()
	def forward(self, x):
		x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
		x = F.max_pool2d(x_pad, 2, stride=1)
		return x


# Function:
# 	expand H, W using their own elements
# Example(stride=2):
# 	(0, 0, ., .):
# 		origin:
# 			1 2
# 			3 4
# 		changed:
# 			1 1 2 2
# 			1 1 2 2
# 			3 3 4 4
# 			3 3 4 4
# Input:
# 	(batch_size, in_channels, h, w)
# Output:
# 	(batch_size, in_channels, h*stride, w*stride)
class Upsample(nn.Module):
	def __init__(self, stride=2):
		super(Upsample, self).__init__()
		self.stride = stride
	def forward(self, x):
		assert(x.data.dim() == 4)
		B = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		x = x.view(B, C, H, 1, W, 1).expand(B, C, H, self.stride, W, self.stride).contiguous().view(B, C, H*self.stride, W*self.stride)
		return x


# Function:
# 	(B, C, H, W) -> (B, hs*ws*C, H//hs, W//ws)
# 	Get each C using stride that feels like skipping
# Input:
# 	(batch_size, in_channels, h, w)
# Output:
# 	(batch_size, ws*hs*C, H//hs, W//ws)
class Reorg(nn.Module):
	def __init__(self, stride=2):
		super(Reorg, self).__init__()
		self.stride = stride
	def forward(self, x):
		assert(x.data.dim() == 4)
		B = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		assert(H % self.stride == 0)
		assert(W % self.stride == 0)
		w_stride = self.stride
		h_stride = self.stride
		# (B, C, H, W) -> (B, C, H//hs, W//ws, hs, ws)
		x = x.view(B, C, H//h_stride, h_stride, W//w_stride, w_stride).transpose(3, 4).contiguous()
		# (B, C, H//hs, W//ws, hs, ws) -> (B, C, hs*ws, -1)
		x = x.view(B, C, (H//h_stride)*(W//w_stride), h_stride*w_stride).transpose(2, 3).contiguous()
		# (B, C, hs*ws, -1) -> (B, hs*ws, C, H//hs, W//ws)
		x = x.view(B, C, h_stride*w_stride, H//h_stride, W//w_stride).transpose(1, 2).contiguous()
		# (B, hs*ws, C, H//hs, W//ws) -> (B, hs*ws*C, H//hs, W//ws)
		x = x.view(B, h_stride*w_stride*C, H//h_stride, W//w_stride)
		return x


# Function:
# 	compute all elements' average each (H, W)
# Input:
# 	(N, C, H, W)
# Output:
# 	(N, C)
class GlobalAvgPool2d(nn.Module):
	def __init__(self):
		super(GlobalAvgPool2d, self).__init__()
	def forward(self, x):
		N = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		x = F.avg_pool2d(x, (H, W))
		x = x.view(N, C)
		return x


# Function:
# 	for route and shortcut
class EmptyModule(nn.Module):
	def __init__(self):
		super(EmptyModule, self).__init__()
	def forward(self, x):
		return x


# Darknet
# Support route shortcut and reorg
class Darknet(nn.Module):
	def __init__(self, cfgfile):
		super(Darknet, self).__init__()
		self.is_yolo3 = False
		self.blocks = parse_cfg(cfgfile)
		self.losses = []
		self.models = self.create_network(self.blocks)
		# Because of all yolo layers' anchors, num_anchors, anchor_step and num_classes is the same.
		self.loss = self.models[len(self.models)-1]
		self.width = int(self.blocks[0]['width'])
		self.height = int(self.blocks[0]['height'])
		if self.blocks[(len(self.blocks)-1)]['type'] == 'region' or 'yolo':
			self.anchors = self.loss.anchors
			self.num_anchors = self.loss.num_anchors
			self.anchor_step = self.loss.anchor_step
			self.num_classes = self.loss.num_classes
		self.header = torch.IntTensor([0, 0, 0, 0, 0])
		self.seen = 0
		# self.multiGPU = False
	def forward(self, x):
		ind = -2
		is_yolo3 = self.is_yolo3
		yolo_outs = []
		outputs = dict()
		# out_boxes = []
		for block in self.blocks:
			ind += 1
			if block['type'] == 'net':
				continue
			elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
				x = self.models[ind](x)
				outputs[ind] = x
			elif block['type'] == 'route':
				layers = block['layers'].split(',')
				layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
				if len(layers) == 1:
					x = outputs[layers[0]]
					outputs[ind] = x
				elif len(layers) == 2:
					x1 = outputs[layers[0]]
					x2 = outputs[layers[1]]
					x = torch.cat((x1, x2), 1)
					outputs[ind] = x
			elif block['type'] == 'shortcut':
				from_layer = int(block['from'])
				activation = block['activation']
				from_layer = from_layer if from_layer > 0 else from_layer + ind
				x1 = outputs[from_layer]
				x2 = outputs[ind-1]
				x  = x1 + x2
				if activation == 'leaky':
					x = F.leaky_relu(x, 0.1, inplace=True)
				elif activation == 'relu':
					x = F.relu(x, inplace=True)
				outputs[ind] = x
			elif block['type'] == 'region':
				continue
				# if self.training:
				# 	yolo_outs.append(x)
				# 	outputs[ind] = None
				# else:
				# 	boxes = self.models[ind](x)
				# 	out_boxes.append(boxes)
			elif block['type'] == 'yolo':
				# continue
				is_yolo3 = True
				yolo_outs.append(x)
				# if self.training:
				# 	yolo_outs.append(x)
				# 	outputs[ind] = None
				# else:
				# 	boxes = self.models[ind](x)
				# 	out_boxes.append(boxes)
			elif block['type'] == 'cost':
				continue
			else:
				print('[Error]:unkown type %s' % (block['type'])) 
		# if self.training:
		# 	# return self.loss
		# 	if is_yolo3:
		# 		# print('[INFO]:This yolov3 darknet-train...')
		# 		return yolo_outs
		# 	else:
		# 		return x
		# else:
		# 	if is_yolo3:
		# 		# print('[INFO]:This yolov3 darknet-test...')
		# 		return out_boxes
		# 	else:
		# 		return x
		# return x
		if is_yolo3:
			return yolo_outs
		else:
			return x
	# merge conv, bn, leaky, etc.
	def create_network(self, blocks):
		models = nn.ModuleList()
		prev_filters = 3
		out_filters =[]
		prev_stride = 1
		out_strides = []
		conv_id = 0
		# all conv stride = 1
		# So out_strides record sizes reduce in scale
		for block in blocks:
			if block['type'] == 'net':
				prev_filters = int(block['channels'])
				continue
			elif block['type'] == 'convolutional':
				conv_id = conv_id + 1
				batch_normalize = int(block['batch_normalize'])
				filters = int(block['filters'])
				kernel_size = int(block['size'])
				stride = int(block['stride'])
				is_pad = int(block['pad'])
				# Problem:
				# 	kernel_size when odd better?
				pad = (kernel_size-1)//2 if is_pad else 0
				activation = block['activation']
				model = nn.Sequential()
				if batch_normalize:
					model.add_module('conv{}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
					model.add_module('bn{}'.format(conv_id), nn.BatchNorm2d(filters))
				else:
					model.add_module('conv{}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
				if activation == 'leaky':
					model.add_module('leaky{}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
				elif activation == 'relu':
					model.add_module('relu{}'.format(conv_id), nn.ReLU(inplace=True))
				prev_filters = filters
				out_filters.append(prev_filters)
				prev_stride = stride * prev_stride
				out_strides.append(prev_stride)
				models.append(model)
			elif block['type'] == 'maxpool':
				pool_size = int(block['size'])
				stride = int(block['stride'])
				if stride > 1:
					model = nn.MaxPool2d(pool_size, stride)
				else:
					model = MaxPoolStride1()
				out_filters.append(prev_filters)
				prev_stride = stride * prev_stride
				out_strides.append(prev_stride)
				models.append(model)
			# at end in general
			elif block['type'] == 'avgpool':
				model = GlobalAvgPool2d()
				out_filters.append(prev_filters)
				models.append(model)
			elif block['type'] == 'softmax':
				model = nn.Softmax()
				out_strides.append(prev_stride)
				out_filters.append(prev_filters)
				models.append(model)
			# the losses are averaged over observations for each minibatch. 
			elif block['type'] == 'cost':
				if block['_type'] == 'sse':
					model = nn.MSELoss(size_average=True)
				elif block['_type'] == 'L1':
					model = nn.L1Loss(size_average=True)
				elif block['_type'] == 'smooth':
					model = nn.SmoothL1Loss(size_average=True)
				out_filters.append(1)
				out_strides.append(prev_stride)
				models.append(model)
			elif block['type'] == 'reorg':
				stride = int(block['stride'])
				prev_filters = stride * stride * prev_filters
				out_filters.append(prev_filters)
				prev_stride = prev_stride * stride
				out_strides.append(prev_stride)
				models.append(Reorg(stride))
			elif block['type'] == 'upsample':
				stride = int(block['stride'])
				out_filters.append(prev_filters)
				prev_stride = prev_stride // stride
				out_strides.append(prev_stride)
				models.append(Upsample(stride))
			elif block['type'] == 'route':
				layers = block['layers'].split(',')
				ind = len(models)
				layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
				if len(layers) == 1:
					prev_filters = out_filters[layers[0]]
					prev_stride = out_strides[layers[0]]
				elif len(layers) == 2:
					assert(layers[0] == ind - 1)
					prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
					prev_stride = out_strides[layers[0]]
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(EmptyModule())
			# other params used in forward function
			elif block['type'] == 'shortcut':
				ind = len(models)
				prev_filters = out_filters[ind-1]
				out_filters.append(prev_filters)
				prev_stride = out_strides[ind-1]
				out_strides.append(prev_stride)
				models.append(EmptyModule())
			elif block['type'] == 'connected':
				filters = int(block['output'])
				if block['activation'] == 'linear':
					model = nn.Linear(prev_filters, filters)
				elif block['activation'] == 'leaky':
					model = nn.Sequential(
								nn.Linear(prev_filters, filters),
								nn.LeakyReLU(0.1, inplace=True))
				elif block['activation'] == 'relu':
					model = nn.Sequential(
								nn.Linear(prev_filters, filters),
								nn.ReLU(inplace=True))
				prev_filters = filters
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(model)
			# Note:
			# 	some values aren't assigned to class
			# 	(RegionLoss and YoloLayer())
			elif block['type'] == 'region':
				loss = RegionLoss()
				anchors = block['anchors'].split(',')
				loss.anchors = [float(i) for i in anchors]
				loss.num_classes = int(block['classes'])
				loss.num_anchors = int(block['num'])
				loss.anchor_step = len(loss.anchors) // loss.num_anchors
				loss.object_scale = float(block['object_scale'])
				loss.noobject_scale = float(block['noobject_scale'])
				loss.class_scale = float(block['class_scale'])
				loss.coord_scale = float(block['coord_scale'])
				loss.stride = prev_stride
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(loss)
				self.losses.append(loss)
			elif block['type'] == 'yolo':
				self.is_yolo3 = True
				yolo_layer = YoloLayer()
				anchors = block['anchors'].split(',')
				anchor_mask = block['mask'].split(',')
				yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
				yolo_layer.anchors = [float(i) for i in anchors]
				yolo_layer.num_classes = int(block['classes'])
				yolo_layer.num_anchors = int(block['num'])
				yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
				yolo_layer.stride = prev_stride
				# yolo_layer.object_scale = float(block['object_scale'])
				# yolo_layer.noobject_scale = float(block['noobject_scale'])
				# yolo_layer.class_scale = float(block['class_scale'])
				# yolo_layer.coord_scale = float(block['coord_scale'])
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(yolo_layer)
				self.losses.append(yolo_layer)
			else:
				print('[Error]:unkown type %s' % (block['type']))
		return models
	def print_network(self):
		print_cfg(self.blocks)
	def load_weights(self, weightfile):
		fp = open(weightfile, 'rb')
		header = np.fromfile(fp, count=5, dtype=np.int32)
		# header = np.fromfile(fp, count=4, dtype=np.int32)
		self.header = torch.from_numpy(header)
		self.seen = self.header[3]
		# print(self.header)
		# print(self.seen)
		buf = np.fromfile(fp, dtype=np.float32)
		# print(len(buf))
		fp.close()
		start = 0
		ind = -2
		for block in self.blocks:
			if start >= buf.size:
				break
			ind = ind + 1
			if block['type'] == 'net':
				continue
			elif block['type'] == 'convolutional':
				model = self.models[ind]
				batch_normalize = int(block['batch_normalize'])
				if batch_normalize:
					start = load_conv_bn(buf, start, model[0], model[1])
				else:
					start = load_conv(buf, start, model[0])
			elif block['type'] == 'connected':
				model = self.models[ind]
				if block['activation'] != 'linear':
					start = load_fc(buf, start, model[0])
				else:
					start = load_fc(buf, start, model)
			# Easier to develop
			elif block['type'] == 'maxpool':
				continue
			elif block['type'] == 'reorg':
				continue
			elif block['type'] == 'upsample':
				continue
			elif block['type'] == 'route':
				continue
			elif block['type'] == 'shortcut':
				continue
			elif block['type'] == 'region':
				continue
			elif block['type'] == 'yolo':
				continue
			elif block['type'] == 'avgpool':
				continue
			elif block['type'] == 'softmax':
				continue
			elif block['type'] == 'cost':
				continue
			else:
				print('[Error]:unkown type %s' % (block['type']))
	# cutoff:
	# 	whether delete end block or not
	def save_weights(self, outfile, cutoff=0):
		if cutoff <= 0:
			cutoff = len(self.blocks) - 1
		fp = open(outfile, 'wb')
		self.header[3] = self.seen
		header = self.header
		header.numpy().tofile(fp)
		ind = -1
		for blockId in range(1, cutoff+1):
			ind = ind + 1
			block = self.blocks[blockId]
			if block['type'] == 'convolutional':
				model = self.models[ind]
				batch_normalize = int(block['batch_normalize'])
				if batch_normalize:
					save_conv_bn(fp, model[0], model[1])
				else:
					save_conv(fp, model[0])
			# activation fun(like ReLU) params don't need be saved.
			elif block['type'] == 'connected':
				model = self.models[ind]
				if block['activation'] != 'linear':
					save_fc(fc, model[0])
				else:
					save_fc(fc, model)
			# Easier to develop
			elif block['type'] == 'maxpool':
				continue
			elif block['type'] == 'reorg':
				continue
			elif block['type'] == 'upsample':
				continue
			elif block['type'] == 'route':
				continue
			elif block['type'] == 'shortcut':
				continue
			elif block['type'] == 'region':
				continue
			elif block['type'] == 'yolo':
				continue
			elif block['type'] == 'avgpool':
				continue
			elif block['type'] == 'softmax':
				continue
			elif block['type'] == 'cost':
				continue
			else:
				print('[Error]:unkown type %s' % (block['type']))
		fp.close()


if __name__ == '__main__':
	# x = torch.Tensor(2, 2, 4, 4).fill_(1)
	# x = nn.Parameter(x)
	# print(x)
	# print(GlobalAvgPool2d()(x))
	d = Darknet('../cfg/me/darknet19_wfc_face.cfg')
	# d = Darknet('../cfg/me/darknet19_wfc_face.cfg')
	d.print_network()
	d.load_weights('../weights/darknet19_wfc.weights')
	# d.load_weights(r'E:\GraduationProject\experiment_record\detection\yolov3_CelebA\train\weights\000042.weights')
	# d.load_weights('../weights/000050.weights')