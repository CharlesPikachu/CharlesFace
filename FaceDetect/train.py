# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
from nets.darknet import Darknet
from utils.standard_utils import *
from config import Settings
import dataset
# from loss_module.region_loss import RegionLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import os
import time


# Config
# --------------------------------------------------------------------------------------------------------------------
# Training Settings
datacfg = Settings.datacfg
cfgfile = Settings.cfgfile
weightfile = Settings.weightfile
data_options = Settings.data_options
net_options = Settings.net_options
trainlist = Settings.trainlist
testlist = Settings.testlist
# testlist = './WFC_train.txt'
backupdir = Settings.backupdir
nsamples = Settings.nsamples
gpus = Settings.gpus
ngpus = Settings.ngpus
num_workers = Settings.num_workers
batch_size = Settings.batch_size
max_batches = Settings.max_batches
learning_rate = Settings.learning_rate
momentum = Settings.momentum
# used in weight decay (L2 penalty)
decay = Settings.decay
# processed_batches num for adjust lr
steps = Settings.steps
# the scale for adjust lr.(lr = lr * scale)
scales = Settings.scales
# whether yolov3 or not
YOLOv3 = Settings.YOLOv3

# Train HParams
max_epochs = Settings.max_epochs
use_cuda = Settings.use_cuda
seed = Settings.seed
eps = Settings.eps
# epoches
save_interval = Settings.save_interval
# batches
dot_interval = Settings.dot_interval

# Test HParams
conf_thresh = Settings.conf_thresh
nms_thresh = Settings.nms_thresh
iou_thresh = Settings.iou_thresh
# --------------------------------------------------------------------------------------------------------------------


# Some pre-process
# --------------------------------------------------------------------------------------------------------------------
if not os.path.exists(backupdir):
	os.mkdir(backupdir)
# Sets the seed for generating random numbers. And returns a torch._C.Generator object.
torch.manual_seed(seed)
if use_cuda:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpus
	torch.cuda.manual_seed(seed)
model = Darknet(cfgfile)
model.print_network()
# Different yolo version
if YOLOv3:
	yolo_loss = model.losses
else:
	region_loss = model.loss
try:
	model.load_weights(weightfile)
	print('[INFO]:%s loaded...' % weightfile)
except:
	pass
model.is_yolov3 = YOLOv3
if YOLOv3:
	for yl in yolo_loss:
		yl.seen = model.seen
else:
	region_loss.seen = model.seen
processed_batches = model.seen // batch_size
init_width = model.width
init_height = model.height
init_epoch = model.seen // nsamples
# pin_memory: copy tensors into CUDA.
# num_workers: subprocesses num whice used for data loading.
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
# just load once.
test_loader = torch.utils.data.DataLoader(
	dataset.listDataset(testlist, 
						shape=(init_width, init_height),
						shuffle=False,
						transform=transforms.Compose([transforms.ToTensor(),]),
						train=False),
	batch_size=batch_size,
	shuffle=False,
	**kwargs)
if use_cuda:
	if ngpus > 1:
		model = nn.DataParallel(model).cuda()
	else:
		model = model.cuda()
# params_dict = dict(model.named_parameters())
# params = []
# for key, value in params_dict.items():
# 	if key.find('.bn') >= 0 or key.find('.bias') >= 0:
# 		params += [{'params': [value], 'weight_decay': 0.0}]
# 	else:
# 		params += [{'params': [value], 'weight_decay': decay*batch_size}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
# --------------------------------------------------------------------------------------------------------------------


# Train and Test
# --------------------------------------------------------------------------------------------------------------------
# adjust lr while training
def adjust_learning_rate(optimizer, batch):
	lr = learning_rate
	for i in range(len(steps)):
		scale = scales[i] if i < len(scales) else 1
		if batch >= steps[i]:
			lr = lr * scale
			if batch == steps[i]:
				break
		else:
			break
	for param_group in optimizer.param_groups:
		# Problem
		# 	why lr by batch_size
		param_group['lr'] = lr/batch_size
	return lr


# Train
def train(epoch):
	global processed_batches
	t0 = time.time()
	if ngpus > 1:
		cur_model = model.module
	else:
		cur_model = model
	train_loader = torch.utils.data.DataLoader(
		dataset.listDataset(trainlist,
							shape=(init_width, init_height),
							shuffle=True,
							transform=transforms.Compose([transforms.ToTensor(),]),
							train=True,
							seen=cur_model.seen,
							batch_size=batch_size,
							num_workers=num_workers),
		batch_size=batch_size,
		shuffle=False,
		**kwargs)
	lr = adjust_learning_rate(optimizer, processed_batches)
	logging('epoch %d, processed %d samples, lr %f, processed_batches %f' % (epoch, epoch * len(train_loader.dataset), lr, processed_batches))
	model.train()
	t1 = time.time()
	avg_time = torch.zeros(9)
	for batch_idx, (data, target) in enumerate(train_loader):
		t2 = time.time()
		adjust_learning_rate(optimizer, processed_batches)
		processed_batches = processed_batches + 1
		if use_cuda:
			data = data.cuda()
		t3 = time.time()
		data, target = Variable(data), Variable(target)
		t4 = time.time()
		optimizer.zero_grad()
		t5 = time.time()
		output = model(data)
		t6 = time.time()
		if YOLOv3:
			for yl in yolo_loss:
				yl.seen = yl.seen + data.data.size(0)
			loss = 0
			assert len(yolo_loss) == len(output)
			for i in range(len(yolo_loss)):
				yl = yolo_loss[i]
				ot = output[i]
				loss += yl(ot, target)
				# loss = yl(ot, target)
				# if i < len(yolo_loss) - 1:
				# 	loss.backward(retain_graph=True)
				# else:
				# 	loss.backward()
			if batch_idx == 0:
				pre_loss = int(loss.data)
			try:
				if pre_loss * 8 < int(loss.data):
					while pre_loss * 8 < int(loss.data):
						loss = loss / 1.2
				else:
					pre_loss = int(loss.data)
			except:
				print('[Warning]:Appear NaN, ignore it by default...')
				loss = 0
				continue
		else:
			region_loss.seen = region_loss.seen + data.data.size(0)
			loss = region_loss(output, target)
			if batch_idx == 0:
				pre_loss = int(loss.data)
			try:
				if pre_loss * 3 < int(loss.data):
					while pre_loss * 3 < int(loss.data):
						loss = loss / 1.01
				else:
					pre_loss = int(loss.data)
			except:
				print('[Warning]:Appear NaN, ignore it by default...')
				loss = 0
				continue
			# print('loss:%s' % loss)
			# t7 = time.time()
			# loss.backward()
			# t8 = time.time()
		t7 = time.time()
		loss.backward()
		t8 = time.time()
		# update the params
		optimizer.step()
		t9 = time.time()
		if False and batch_idx > 1:
			avg_time[0] = avg_time[0] + (t2 - t1)
			avg_time[1] = avg_time[1] + (t3 - t2)
			avg_time[2] = avg_time[2] + (t4 - t3)
			avg_time[3] = avg_time[3] + (t5 - t4)
			avg_time[4] = avg_time[4] + (t6 - t5)
			avg_time[5] = avg_time[5] + (t7 - t6)
			avg_time[6] = avg_time[6] + (t8 - t7)
			avg_time[7] = avg_time[7] + (t9 - t8)
			avg_time[8] = avg_time[8] + (t9 - t1)
			print('-------------------------------')
			print('       load data : %f' % (avg_time[0] / (batch_idx)))
			print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
			print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
			print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
			print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
			print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
			print('        backward : %f' % (avg_time[6] / (batch_idx)))
			print('            step : %f' % (avg_time[7] / (batch_idx)))
			print('           total : %f' % (avg_time[8] / (batch_idx)))
		t1 = time.time()
		# if batch_idx % dot_interval == 0 and batch_idx > 1:
		# 	cur_model.save_weights('%s/temp.weights' % backupdir)
	t1 = time.time()
	logging('training with %f samples/s' % (len(train_loader.dataset) / (t1-t0)))
	if ((epoch + 1) % save_interval == 0) or ((epoch + 1) == max_epochs):
		logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
		cur_model.seen = (epoch + 1) * len(train_loader.dataset)
		cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))


# Test while training
def test(epoch):
	def truths_length(truths):
		# Max objects in one picture
		for i in range(50):
			if truths[i][1] == 0:
				return i
	model.eval()
	if ngpus > 1:
		cur_model = model.module
	else:
		cur_model = model
	yolo_layers = cur_model.losses
	num_classes = cur_model.num_classes
	anchors = cur_model.anchors
	if not YOLOv3:
		anchors = [anchor/cur_model.loss.stride for anchor in anchors]
	num_anchors = cur_model.num_anchors
	total = 0.0
	# num of box that det_conf > conf_thresh
	proposals = 0.0
	# num of box that det_conf > conf_thresh and cls_conf > iou_thresh
	correct = 0.0
	for batch_idx, (data, target) in enumerate(test_loader):
		if use_cuda:
			data = data.cuda()
		data = Variable(data, volatile=True)
		if YOLOv3:
			# print('[INFO]:This is yolov3 train-test...')
			all_boxes = []
			output = model(data)
			assert len(yolo_layers) == len(output)
			for i in range(len(yolo_layers)):
				yl = yolo_layers[i]
				yl.thresh = conf_thresh
				op = output[i]
				boxes = yl(op)
				all_boxes.append(boxes)
			# all_boxes = model(data)
			layers_num = len(all_boxes)
			batches_num = len(all_boxes[0])
		else:
			output = model(data).data
			all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
			batches_num = output.size(0)
		for bnum in range(batches_num):
			# pre-defined for yolov3
			boxes = []
			if YOLOv3:
				# print('[INFO]:This is yolov3 train-test...')
				for lnum in range(layers_num):
					boxes += all_boxes[lnum][bnum]
			else:
				boxes = all_boxes[bnum]
			boxes = nms(boxes, nms_thresh)
			# classes, x, y, w, h
			truths = target[bnum].view(-1, 5)
			num_gts = truths_length(truths)
			total += num_gts
			for j in range(len(boxes)):
				if boxes[j][4] > conf_thresh:
					proposals += 1
			for k in range(num_gts):
				# [x, y, w, h, det_conf, cls_max_conf, cls_max_id]
				box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
				best_iou = 0
				best_kk = -1
				for kk in range(len(boxes)):
					iou = bbox_iou(box_gt, boxes[kk], x1y1x2y2=False)
					if iou > best_iou:
						best_kk = kk
						best_iou = iou
				if best_iou > iou_thresh and boxes[best_kk][6] == box_gt[6]:
					correct += 1
	precision = 1.0 * correct / (proposals + eps)
	recall = 1.0 * correct / (total + eps)
	fscore = 2.0 * precision * recall / (precision + recall + eps)
	print(correct, proposals, total)
	logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
# --------------------------------------------------------------------------------------------------------------------




if __name__ == '__main__':
	# test(0)
	# train(19)
	for epoch in range(init_epoch, max_epochs):
		train(epoch)
		if (epoch + 1) % (save_interval + 2)== 0:
		# if (epoch + 1) % save_interval== 0:
			error_num = 0
			while True:
				try:
					test(epoch)
					break
				except:
					error_num += 1
					print('[Error]: test error, try again...')
					if error_num > 3:
						break
					continue