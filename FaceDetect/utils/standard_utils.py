# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import math
import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import os


# Output time now + message
def logging(message):
	print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


# Sigmoid
def sigmoid(x):
	return 1.0/(math.exp(-x)+1.)


# Softmax
def softmax(x):
	x = torch.exp(x - torch.max(x))
	x = x / x.sum()
	return x


# GPU -> CPU
# Float
def convert2cpu(gpu_matrix):
	return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
# Long
def convert2cpu_long(gpu_matrix):
	return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


# Function:
# 	compute IOU
# 	IOU = Overlapping area / (two boxes' total area - Overlapping area)
# 	Max: 1, Min: 0
# x1y1x2y2:
# 	True: (x1, y1, x2, y2)
# 	False: (x, y, w, h)
# many
def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
	if x1y1x2y2:
		mx = torch.min(boxes1[0], boxes2[0])
		Mx = torch.max(boxes1[2], boxes2[2])
		my = torch.min(boxes1[1], boxes2[1])
		My = torch.max(boxes1[3], boxes2[3])
		w1 = boxes1[2] - boxes1[0]
		h1 = boxes1[3] - boxes1[1]
		w2 = boxes2[2] - boxes2[0]
		h2 = boxes2[3] - boxes2[1]
	else:
		mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
		Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
		my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
		My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
		w1 = boxes1[2]
		h1 = boxes1[3]
		w2 = boxes2[2]
		h2 = boxes2[3]
	uw = Mx - mx
	uh = My - my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	mask = ((cw <= 0) + (ch <= 0) > 0)
	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw * ch
	carea[mask] = 0
	uarea = area1 + area2 - carea
	return carea/uarea
# one
def bbox_iou(box1, box2, x1y1x2y2=True):
	if x1y1x2y2:
		mx = min(box1[0], box2[0])
		Mx = max(box1[2], box2[2])
		my = min(box1[1], box2[1])
		My = max(box1[3], box2[3])
		w1 = box1[2] - box1[0]
		h1 = box1[3] - box1[1]
		w2 = box2[2] - box2[0]
		h2 = box2[3] - box2[1]
	else:
		mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
		Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
		my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
		My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
		w1 = box1[2]
		h1 = box1[3]
		w2 = box2[2]
		h2 = box2[3]
	uw = Mx - mx
	uh = My - my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	carea = 0
	if cw <= 0 or ch <= 0:
		return 0.0
	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw * ch
	uarea = area1 + area2 - carea
	try:
		return carea/uarea
	except:
		return 0.0


# validation:
# 	whether predict multi-classes in one box(all > thresh).
# conf_thresh:
# 	conf = cls_conf * loc_conf
# only_object:
# 	one box only have one object(if True).
def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
	anchor_step = len(anchors) // num_anchors
	# One picture
	if output.dim() == 3:
		output = output.unsqueeze(0)
	batch = output.size(0)
	# make sure the size of output.
	assert(output.size(1) == (5+num_classes)*num_anchors)
	h = output.size(2)
	w = output.size(3)
	# Part1
	t0 = time.time()
	all_boxes = []
	# Note:
	# 	Must correspond to train process
	# (batch, (5+num_classes)*num_anchors, h, w) -> 
	# (batch*num_anchors, 5+num_classes, h*w) ->
	# (5+num_classes, batch*num_anchors, h*w) ->
	# (5+num_classes, batch*num_anchors*h*w)
	output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0, 1).contiguous().view(5+num_classes, batch*num_anchors*h*w)
	# (w) -> (h, w) -> (b*num_anchors, h, w) -> (batch*num_anchors*h*w)
	# type_as:
	# 	cuda
	grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output)
	# (h) -> (w, h) -> (h, w) -> (batch*num_anchors, h, w) -> (batch*num_anchors*h*w)
	# type_as:
	# 	cuda
	grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output)
	xs = torch.sigmoid(output[0]) + grid_x
	ys = torch.sigmoid(output[1]) + grid_y
	anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
	anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
	# type_as:
	# 	cuda
	# (num_anchors, 1) -> (batch*num_anchors, 1) -> (1, batch*num_anchors, h*w) -> (batch*num_anchors*h*w)
	anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output)
	anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output)
	ws = torch.exp(output[2]) * anchor_w
	hs = torch.exp(output[3]) * anchor_h
	# detect confidence
	det_confs = torch.sigmoid(output[4])
	# class confidence
	# (num_classes, batch*num_anchors*h*w) -> (batch*num_anchors*h*w, num_classes)
	cls_confs = torch.nn.Softmax(dim=1)(Variable(output[5: 5+num_classes].transpose(0, 1))).data
	cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
	cls_max_confs = cls_max_confs.view(-1)
	cls_max_ids = cls_max_ids.view(-1)
	t1 = time.time()
	# Part2
	sz_hw = h * w
	sz_hwa = sz_hw * num_anchors
	det_confs = convert2cpu(det_confs)
	cls_max_confs = convert2cpu(cls_max_confs)
	cls_max_ids = convert2cpu_long(cls_max_ids)
	xs = convert2cpu(xs)
	ys = convert2cpu(ys)
	ws = convert2cpu(ws)
	hs = convert2cpu(hs)
	if validation:
		cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
	t2 = time.time()
	# Part3
	for b in range(batch):
		boxes = []
		for cy in range(h):
			for cx in range(w):
				for i in range(num_anchors):
					ind = b*sz_hwa + i*sz_hw + cy*w + cx
					det_conf =  det_confs[ind]
					if only_objectness:
						conf =  det_confs[ind]
					else:
						# define refer to paper
						conf = det_confs[ind] * cls_max_confs[ind]
					if conf > conf_thresh:
						bcx = xs[ind]
						bcy = ys[ind]
						bw = ws[ind]
						bh = hs[ind]
						cls_max_conf = cls_max_confs[ind]
						cls_max_id = cls_max_ids[ind]
						# convert to the scale relative to origin image
						box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
						if (not only_objectness) and validation:
							for c in range(num_classes):
								tmp_conf = cls_confs[ind][c]
								if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
									box.append(tmp_conf)
									box.append(c)
						boxes.append(box)
		all_boxes.append(boxes)
	t3 = time.time()
	# Part4
	if False:
		print('---------------------------------')
		print('matrix computation : %f' % (t1-t0))
		print('        gpu to cpu : %f' % (t2-t1))
		print('      boxes filter : %f' % (t3-t2))
		print('---------------------------------')
	return all_boxes


# plot boxes in cv2 Demo.
# cv2 image or skimage.
def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
	import cv2
	colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
	# c, offset, classes
	def get_color(c, x, max_val):
		# [0, 5)
		ratio = float(x) / max_val * 5
		# [0, 1, 2, 3, 4]
		i = int(math.floor(ratio))
		# [1, 2, 3, 4, 5]
		j = int(math.ceil(ratio))
		# [0, 1)
		ratio = ratio - i
		r = (1-ratio) * colors[i][c] + ratio * colors[j][c]
		return int(r*255)
	try:
		# print('[INFO]: Cv2 image(same)...')
		width = img.shape[1]
		height = img.shape[0]
	except:
		print('[Warning]: Others(not same)...')
		exit(-1)
	for i in range(len(boxes)):
		box = boxes[i]
		# correspond to function "get_region_boxes"
		# results rounding
		# left top
		x1 = int(round((box[0] - box[2]/2.0) * width, 0))
		y1 = int(round((box[1] - box[3]/2.0) * height, 0))
		# right bottom
		x2 = int(round((box[0] + box[2]/2.0) * width, 0))
		y2 = int(round((box[1] + box[3]/2.0) * height, 0))
		if color:
			rgb = color
		else:
			rgb = (255, 0, 0)
		# correspond to function "get_region_boxes"
		if len(box) >= 7 and class_names:
			cls_conf = box[5]
			cls_id = box[6]
			print('%s: %f' % (class_names[cls_id], cls_conf))
			classes = len(class_names)
			offset = cls_id * 123457 % classes
			# choose box color
			red = get_color(2, offset, classes)
			green = get_color(1, offset, classes)
			blue = get_color(0, offset, classes)
			if color is None:
				rgb = (red, green, blue)
			# picture, add_text, left top coord, font, font_size, color, font thickness
			img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
		# pic, left top, right bottom, color, line class
		img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
	if savename:
		print("[INFO]:save plot results to %s" % savename)
		cv2.imwrite(savename, img)
	return img


# plot all boxes in one image.
def plot_boxes(img, boxes, savename=None, class_names=None):
	colors = torch.FloatTensor([[1, 0, 1],[0, 0, 1],[0, 1, 1],[0, 1, 0],[1, 1, 0],[1, 0, 0]])
	def get_color(c, x, max_val):
		ratio = float(x)/max_val * 5
		i = int(math.floor(ratio))
		j = int(math.ceil(ratio))
		ratio = ratio - i
		r = (1-ratio) * colors[i][c] + ratio * colors[j][c]
		return int(r*255)
	try:
		# print('[INFO]: PIL image(same)...')
		width = img.width
		height = img.height
	except:
		print('[Warning]: Others(not same)...')
		exit(-1)
	draw = ImageDraw.Draw(img)
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = (box[0] - box[2]/2.0) * width
		y1 = (box[1] - box[3]/2.0) * height
		x2 = (box[0] + box[2]/2.0) * width
		y2 = (box[1] + box[3]/2.0) * height
		rgb = (255, 0, 0)
		if len(box) >= 7 and class_names:
			cls_conf = box[5]
			cls_id = box[6]
			# print(cls_id)
			print('%s: %f' % (class_names[cls_id], cls_conf))
			classes = len(class_names)
			offset = cls_id * 123457 % classes
			red = get_color(2, offset, classes)
			green = get_color(1, offset, classes)
			blue = get_color(0, offset, classes)
			rgb = (red, green, blue)
			draw.text((x1+5, y1+5), class_names[cls_id], fill=rgb)
			# draw.text((x1, y1), class_names[cls_id], fill=rgb)
		# draw.rectangle([x1, y1, x2, y2], outline=rgb)
		w = box[2] * width
		h = box[3] * height
		draw.line([(x1, y1), (x1+w, y1), (x1+w, y1+h), (x1, y1+h), (x1, y1)], width=5, fill=rgb)
	if savename:
		print("[INFO]:save plot results to %s" % savename)
		img.save(savename)
	return img


# non maximum suppression
def nms(boxes, nms_thresh):
	if len(boxes) == 0:
		return boxes
	det_confs = torch.zeros(len(boxes))
	for i in range(len(boxes)):
		det_confs[i] = boxes[i][4]
	# the smaller index, the better detect confidence.
	_, sortIds = torch.sort(det_confs, descending=True)
	out_boxes = []
	for i in range(len(boxes)):
		box_i = boxes[sortIds[i]]
		if box_i[4] > 0:
			out_boxes.append(box_i)
			for j in range(i+1, len(boxes)):
				box_j = boxes[sortIds[j]]
				if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
					box_j[4] = 0
	return out_boxes


# PIL image convert to Torch
def image2torch(img):
	width = img.width
	height = img.height
	img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
	# (height, width, 3) -> (width, height, 3) -> (3, width, height)
	img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
	# (3, width, height) -> (1, 3, height, width)
	img = img.view(1, 3, height, width)
	img = img.float().div(255.0)
	return img


# detect of yolov3.
# Only for one picture now.
def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
	model.eval()
	# Part1
	t0 = time.time()
	# PIL image
	if isinstance(img, Image.Image):
		img = image2torch(img)
	# cv2 image
	elif type(img) == np.ndarray:
		img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
	else:
		print("[Error]: unknow image type...")
		exit(-1)
	t1 = time.time()
	# Part2
	if use_cuda:
		img = img.cuda()
	img = torch.autograd.Variable(img)
	t2 = time.time()
	# Part3
	# list_boxes = model(img)
	yolo_layers = model.losses
	output = model(img)
	assert len(yolo_layers) == len(output)
	list_boxes = []
	for i in range(len(yolo_layers)):
		yl = yolo_layers[i]
		yl.thresh = conf_thresh
		op = output[i]
		temp = yl(op)
		list_boxes.append(temp)
	# Because of three 'yolo' layers now.
	# So you can also use(quicker than using way.):
	# 	boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
	# use generic statement
	boxes = []
	for i in range(len(list_boxes)):
		boxes += list_boxes[i][0]
	t3 = time.time()
	# Part4
	boxes = nms(boxes, nms_thresh)
	t4 = time.time()
	# Part5
	if False:
		print('-----------------------------------')
		print(' image to tensor : %f' % (t1 - t0))
		print('  tensor to cuda : %f' % (t2 - t1))
		print('         predict : %f' % (t3 - t2))
		print('             nms : %f' % (t4 - t3))
		print('           total : %f' % (t4 - t0))
		print('-----------------------------------')
	return boxes


# load class names
def load_class_names(namesfile):
	class_names = []
	with open(namesfile, 'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.rstrip()
		class_names.append(line)
	return class_names


# resize bboxes
def scale_bboxes(bboxes, width, height):
	import copy
	dets = copy.deepcopy(bboxes)
	for i in range(len(dets)):
		dets[i][0] = dets[i][0] * width
		dets[i][1] = dets[i][1] * height
		dets[i][2] = dets[i][2] * width
		dets[i][3] = dets[i][3] * height
	return dets


# refer: https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch04s07.html
# count file lines(the number of \n)
def file_lines(thefilepath):
	count = 0
	thefile = open(thefilepath, 'rb')
	while True:
		buffer = thefile.read(8192*1024)
		if not buffer:
			break
		try:
			count += buffer.count('\n')
		except:
			count += buffer.count(b'\n')
	thefile.close()
	# count = len(open(thefilepath).readlines())
	# for line in open(thefilepath).xreadlines(): count += 1
	return count


# refer: https://stackoverflow.com/questions/8032642/how-to-obtain-image-size-using-standard-python-class-without-using-external-lib
# Determine the image type of fhandle and return its size. from draco
def get_image_size(fname):
	import struct
	import imghdr
	with open(fname, 'rb') as fhandle:
		head = fhandle.read(24)
		if len(head) != 24: 
			return
		if imghdr.what(fname) == 'png':
			check = struct.unpack('>i', head[4: 8])[0]
			if check != 0x0d0a1a0a:
				return
			width, height = struct.unpack('>ii', head[16: 24])
		elif imghdr.what(fname) == 'gif':
			width, height = struct.unpack('<HH', head[6: 10])
		elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
			try:
				# Read 0xff next
				fhandle.seek(0)
				size = 2
				ftype = 0
				while not 0xc0 <= ftype <= 0xcf:
					fhandle.seek(size, 1)
					byte = fhandle.read(1)
					while ord(byte) == 0xff:
						byte = fhandle.read(1)
					ftype = ord(byte)
					size = struct.unpack('>H', fhandle.read(2))[0] - 2
				# We are at a SOFn block
				# Skip `precision' byte.
				fhandle.seek(1, 1)
				height, width = struct.unpack('>HH', fhandle.read(4))
			#IGNORE:W0703
			except Exception:
				return
		else:
			return
		return width, height


# read ".data" file in "cfg" folder
def read_data_cfg(datacfg):
	options = dict()
	options['gpus'] = '0,1,2,3'
	options['num_workers'] = '10'
	with open(datacfg, 'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.strip()
		if line == '':
			continue
		key, value = line.split('=')
		key = key.strip()
		value = value.strip()
		options[key] = value
	return options


# read truths data
# output:
# 	type(np.array([]))
def read_truths(lab_path):
	if not os.path.exists(lab_path):
		return np.array([])
	if os.path.getsize(lab_path):
		truths = np.loadtxt(lab_path)
		# avoid single truth problem
		truths = truths.reshape(truths.size//5, 5)
		return truths
	else:
		return np.array([])


# read truths(changed label) args
# min_box_scale:
# 	8.0/img.width
def read_truths_args(lab_path, min_box_scale):
	truths = read_truths(lab_path)
	new_truths = []
	for i in range(truths.shape[0]):
		# class, x, y, w, h
		if truths[i][3] < min_box_scale:
			continue
		new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
	return np.array(new_truths)




if __name__ == '__main__':
	print(get_image_size('eagle.png'))