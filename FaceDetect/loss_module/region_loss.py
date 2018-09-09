# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append('../')
from utils.standard_utils import *


# target:
# 	(batch_size, 50*5)
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
	nB = target.size(0)
	nA = num_anchors
	nC = num_classes
	anchor_step = len(anchors) // num_anchors
	conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
	coord_mask = torch.zeros(nB, nA, nH, nW)
	cls_mask = torch.zeros(nB, nA, nH, nW)
	tx = torch.zeros(nB, nA, nH, nW)
	ty = torch.zeros(nB, nA, nH, nW)
	tw = torch.zeros(nB, nA, nH, nW)
	th = torch.zeros(nB, nA, nH, nW)
	tconf = torch.zeros(nB, nA, nH, nW)
	tcls = torch.zeros(nB, nA, nH, nW)
	# Anchor number each batch
	nAnchors = nA*nH*nW
	# Pixel number each feature map
	nPixels  = nH*nW
	for b in range(nB):
		# pre_boxes: (nB*nA*nH*nW, 4)
		# cur_pred_boxes: (nAnchors, 4) -> (4, nAnchors)
		cur_pred_boxes = pred_boxes[b*nAnchors: (b+1)*nAnchors].t()
		cur_ious = torch.zeros(nAnchors)
		# kind num < 50 in each picture.
		for t in range(50):
			if target[b][t*5+1] == 0:
				break
			gx = target[b][t*5+1] * nW
			gy = target[b][t*5+2] * nH
			gw = target[b][t*5+3] * nW
			gh = target[b][t*5+4] * nH
			# current ground truth boxes
			# (4) -> (nAnchors, 4) -> (4, nAnchors)
			cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
			# Max IOU of Each pred box and ground truth box
			# pred_boxes best match one of the ground truth boxes
			cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
		conf_mask[b][cur_ious>sil_thresh] = 0
	# Problem:
	# 	Why seen should smaller than 12800?
	# One explain:
	# 	Ensure the stability of net.
	if seen < 12800:
		if anchor_step == 4:
			# Problem:
			# 	all w ?
			# (len(anchors)) -> (nA, anchor_step) -> (nA) -> (1, nA, 1, 1) -> (nB, nA, nH, nW)
			tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
			# (len(anchors)) -> (nA, anchor_step) -> (nA) -> (1, nA, 1, 1) -> (nB, nA, nH, nW)
			ty = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([3])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
		else:
			# (nB, nA, nH, nW)
			tx.fill_(0.5)
			ty.fill_(0.5)
		# (nB, nA, nH, nW)
		tw.zero_()
		th.zero_()
		coord_mask.fill_(1)
	# ground truth number
	nGT = 0
	# correct number
	nCorrect = 0
	for b in range(nB):
		for t in range(50):
			if target[b][t*5+1] == 0:
				break
			nGT = nGT + 1
			best_iou = 0.0
			# w and h match best
			best_n = -1
			min_dist = 10000
			gx = target[b][t*5+1] * nW
			gy = target[b][t*5+2] * nH
			gi = int(gx)
			gj = int(gy)
			gw = target[b][t*5+3] * nW
			gh = target[b][t*5+4] * nH
			gt_box = [0, 0, gw, gh]
			# Problem:
			# 	cannot understand when (0, 0) anchor_w don't product e^output_w,
			# 	but other cases product.
			for n in range(nA):
				aw = anchors[anchor_step*n]
				ah = anchors[anchor_step*n+1]
				anchor_box = [0, 0, aw, ah]
				iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
				# if ious equal,
				# the coord closer, the better.
				if anchor_step == 4:
					ax = anchors[anchor_step*n+2]
					ay = anchors[anchor_step*n+3]
					dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
				if iou > best_iou:
					best_iou = iou
					best_n = n
				elif anchor_step == 4 and iou == best_iou and dist < min_dist:
					best_iou = iou
					best_n = n
					min_dist = dist
			gt_box = [gx, gy, gw, gh]
			# pred_boxes: (nB*nA*nH*nW, 4)
			# b*nAnchors + best_n*nPixels + gj*nW + gi
			# b*nAnchors: loop data start, length: nAnchors*nH*nW
			# So, best_n*nPixels confirm nA index,
			# gj*nW + gi confirm specific index
			pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
			# if seen < 12800:
			# 	origin: 1
			# else:
			# 	origin: 0
			coord_mask[b][best_n][gj][gi] = 1
			# origin: 0
			cls_mask[b][best_n][gj][gi] = 1
			# origin: 0 or 1
			conf_mask[b][best_n][gj][gi] = object_scale
			# decimal part of target[b][t*5+1] * nW
			tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
			# decimal part of target[b][t*5+2] * nH
			ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
			# ln(target[b][t*5+3] * nW / anchors[anchor_step*best_n)
			tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
			# ln(target[b][t*5+4] * nH / anchors[anchor_step*best_n+1)
			th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
			# best iou
			iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
			# origin: 0
			tconf[b][best_n][gj][gi] = iou
			# origin: 0
			tcls[b][best_n][gj][gi] = target[b][t*5]
			if iou > 0.5:
				nCorrect += 1
	# nGT: ground truth box number
	# nCorrect: number of best iou, which greater than 0.5,  between gt_box and pred_box 
	# coord_mask: coord mask
	# conf_mask: confidence mask
	# cls_mask: classes mask
	# t...: with best iou
	return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


# compute loss
class RegionLoss(nn.Module):
	def __init__(self, num_classes=0, anchors=[], num_anchors=1):
		super(RegionLoss, self).__init__()
		self.num_classes = num_classes
		self.anchors = anchors
		self.num_anchors = num_anchors
		self.anchor_step = len(anchors) // num_anchors
		self.coord_scale = 1
		self.noobject_scale = 1
		self.object_scale = 5
		self.class_scale = 1
		self.thresh = 0.6
		self.seen = 0
		self.stride = 32
	def forward(self, output, target):
		# print(self.noobject_scale, self.object_scale, self.coord_scale, self.class_scale)
		t0 = time.time()
		# Part1
		# batch_size
		nB = output.data.size(0)
		nA = self.num_anchors
		nC = self.num_classes
		# H
		nH = output.data.size(2)
		# W
		nW = output.data.size(3)
		anchors = [anchor/self.stride for anchor in self.anchors]
		output = output.view(nB, nA, (5+nC), nH, nW)
		# anchor box is decided by (x, y, w, h)
		# 0-3
		x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
		y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
		w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
		h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
		# box confidence
		conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
		# linspace:
		# 	step = (end - start) / (num - 1)
		# 	so here step=1
		cls_ = output.index_select(2, Variable(torch.linspace(5, 5+nC-1, nC).long().cuda()))
		# (nB, nA, nC, nH, nW) -> (nB*nA, nC, nH*nW) -> (nB*nA, nH*nW, nC) -> (nB*nA*nH*nW, nC)
		cls_ = cls_.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(nB*nA*nH*nW, nC)
		t1 = time.time()

		# Part2
		pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
		# (nW) -> (nH, nW) -> (nB*nA, nH, nW) -> (nB*nA*nH*nW)
		grid_x = torch.linspace(0, nW-1, nW).repeat(nH, 1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		# (nH) -> (nW, nH) -> (nH, nW) -> (nB*nA, nH, nW) -> (nB*nA*nH*nW)
		grid_y = torch.linspace(0, nH-1, nH).repeat(nW, 1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		# Anchor_step = 2 or 4
		# transfer anchors to torch.Tensor
		# (len(anchors)) -> (nA, anchor_step) -> (nA)
		anchor_w = torch.Tensor(anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
		# transfer anchors to torch.Tensor
		# (len(anchors)) -> (nA, anchor_step) -> (nA)
		anchor_h = torch.Tensor(anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
		# (nA, 1) -> (nB*nA, 1) -> (1, nA*nB, nH*nW) -> (nB*nA*nH*nW)
		anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		# (nA, 1) -> (nB*nA, 1) -> (1, nA*nB, nH*nW) -> (nB*nA*nH*nW)
		anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		# Because of output_x and output_y is between (0, 1), so add (0, nW) and (0, nH) as pred (x, y)
		pred_boxes[0] = x.data + grid_x
		pred_boxes[1] = y.data + grid_y
		pred_boxes[2] = torch.exp(w.data) * anchor_w
		pred_boxes[3] = torch.exp(h.data) * anchor_h
		# (4, nB*nA*nH*nW) -> (nB*nA*nH*nW, 4)
		pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
		t2 = time.time()

		# Part3
		nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = \
			build_targets(pred_boxes, target.data, anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
		cls_mask = (cls_mask == 1)
		# conf size: (nB, nA, nH, nW)
		# number of conf > 0.25
		nProposals = int((conf > 0.25).sum().data[0])
		# To GPU
		tx = Variable(tx.cuda())
		ty = Variable(ty.cuda())
		tw = Variable(tw.cuda())
		th = Variable(th.cuda())
		tconf = Variable(tconf.cuda())
		# best ious
		tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())
		coord_mask = Variable(coord_mask.cuda())
		# sqrt
		# Problem:
		# 	why sqrt.
		conf_mask = Variable(conf_mask.cuda().sqrt())
		# Note:
		# 	class mask should repeat for cls_
		# (nB, nA, nH, nW) -> (nB*nA*nH*nW, 1) -> (nB*nA*nH*nW, nC)
		cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
		# cls_ size: (nB*nA*nH*nW, nC)
		# cls_mask: (nB*nA*nH*nW, nC)
		# Get best ious' classes prediction
		cls_ = cls_[cls_mask].view(-1, nC)
		t3 = time.time()

		# Part4
		# the losses are summed for each minibatch
		# Complain:
		# 	(1) loss_conf:
		# 		IOU of pred box and ground truth box.
		# 		if IOU > thresh, the loss is zero, accomplished by conf_mask.
		# 		Other case:
		# 			1) have object -> object_scale * iou 
		# 			2) have no object -> noobject_scale * 0
		# 		scale may because of object and noobject num unbalance.
		# 	Purpose: make output conf_pred better.
		# 	(2) loss_cls:
		# 		classes prediction loss.
		# 		t_cls: correct class in box, cls_: pred class in box.
		# 		only the best iou have the loss.
		# 	(3) loss_x, loss_y, loss_w, loss_h:
		# 		pred: output_x, output_y, output_w, output_h.
		# 		target: 
		# 			if seen < 12800:
		# 				1) IOU best: 
		# 					tx, ty: gx - int(gx) and gy - int(gy)
		# 					tw, th: the same as "else" part.
		# 				Purpose: make IoU best boxes closer to ground truth (x, y)
		# 				2) not IOU best:
		# 					tx, ty: (0.5, 0.5) if not assigned, it means in each Raster center.
		# 					tw, th: (0, 0)
		# 			else:
		# 				Only compute the best ious location loss.
		# 				tx, ty: (0, 0)
		# 				tw, th: "ln(w * nW / anchor_w)" Corresponds to "pred_w = e^output_w * anchor_w" when predict.
		# 							w: ground truth w
		# 							nW: picture_width / downsamples
		# 							anchor: k-means predict box
		# 						th is computed the same.
		# 						Problem: can't understand why choose ln, maybe depends on kmeans.
		loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
		loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
		loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
		loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
		loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
		# 0 dim represent anchor box and 1 dim represent the classes' probability for each anchor content
		loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls_, tcls)
		# try to avoid nan.
		loss = 0
		count = 0
		for loss_i in [loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0]]:
			count += 1
			if str(loss_i) == 'nan':
				continue
			else:
				if count == 1:
					assert loss_i == loss_x.data[0]
					loss += loss_x
				elif count == 2:
					assert loss_i == loss_y.data[0]
					loss += loss_y
				elif count == 3:
					assert loss_i == loss_w.data[0]
					loss += loss_w
				elif count == 4:
					assert loss_i == loss_h.data[0]
					loss += loss_h
				elif count == 5:
					assert loss_i == loss_conf.data[0]
					loss += loss_conf
				elif count == 6:
					assert loss_i == loss_cls.data[0]
					loss += loss_cls
		# loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
		t4 = time.time()

		# Part5
		if False:
			print('-----------------------------------')
			print('        activation : %f' % (t1 - t0))
			print(' create pred_boxes : %f' % (t2 - t1))
			print('     build targets : %f' % (t3 - t2))
			print('       create loss : %f' % (t4 - t3))
			print('             total : %f' % (t4 - t0))
		print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
		return loss