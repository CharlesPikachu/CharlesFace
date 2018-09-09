# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
# Function:
# 	The same as test fun in train.py.
import torch
import time
from torchvision import transforms
import dataset
from config import Settings
import os
# from loss_module.region_loss import RegionLoss
import torch.nn as nn
from torch.autograd import Variable
from nets.darknet import Darknet
from utils.standard_utils import *


# config
# --------------------------------------------------------------------------------------------------------------------
# Settings
cfgfile = Settings.cfgfile
trainlist = Settings.trainlist
# testlist = Settings.testlist
testlist = './WFC_train.txt'
gpus = Settings.gpus
ngpus = Settings.ngpus
num_workers = Settings.num_workers
batch_size = Settings.batch_size
YOLOv3 = Settings.YOLOv3
weightfile = Settings.weightfile
# train
use_cuda = Settings.use_cuda
seed = Settings.seed
eps = Settings.eps
# test
conf_thresh = Settings.conf_thresh
nms_thresh = Settings.nms_thresh
iou_thresh = Settings.iou_thresh
# --------------------------------------------------------------------------------------------------------------------


# some pre-process
# --------------------------------------------------------------------------------------------------------------------
torch.manual_seed(seed)
if use_cuda:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpus
	torch.cuda.manual_seed(seed)
model = Darknet(cfgfile)
model.print_network()
model.is_yolov3 = YOLOv3
init_width = model.width
init_height = model.height
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
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
# --------------------------------------------------------------------------------------------------------------------


# Test
# --------------------------------------------------------------------------------------------------------------------
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
	if ngpus > 1:
		model.module.load_weights(weightfile)
	else:
		model.load_weights(weightfile)
	logging('evaluating ... %s' % (weightfile))
	test(0)