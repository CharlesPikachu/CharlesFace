# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import random
import os
from PIL import Image
import numpy as np


# im: Image
# c: Image channel
# v: scale
def scale_image_channel(im, c, v):
	cs = list(im.split())
	cs[c] = cs[c].point(lambda i: i * v)
	out = Image.merge(im.mode, tuple(cs))
	return out


# hue: adjust HSV Image hue.
# sat: adjust HSV Image saturation.
# val: adjust HSV Image value.
def distort_image(im, hue, sat, val):
	im = im.convert('HSV')
	# HSV -> hue, saturation, value.
	cs = list(im.split())
	cs[1] = cs[1].point(lambda i: i * sat)
	cs[2] = cs[2].point(lambda i: i * val)
	def change_hue(x):
		x += hue*255
		if x > 255:
			x -= 255
		if x < 0:
			x += 255
		return x
	cs[0] = cs[0].point(change_hue)
	im = Image.merge(im.mode, tuple(cs))
	im = im.convert('RGB')
	return im


# Generate random scale.
def rand_scale(s):
	scale = random.uniform(1, s)
	if(random.randint(1, 10000) % 2):
		return scale
	return 1.0 / scale


# random distort Image in HSV format.
# hue: hue threshold.
# saturation: saturation threshold.
# exposure: value threshold.
def random_distort_image(im, hue, saturation, exposure):
	# [-hue, hue)
	dhue = random.uniform(-hue, hue)
	dsat = rand_scale(saturation)
	dexp = rand_scale(exposure)
	res = distort_image(im, dhue, dsat, dexp)
	return res


# Image augmentation.
def data_augmentation(img, shape, jitter, hue, saturation, exposure):
	oh = img.height
	ow = img.width
	dw = int(ow*jitter)
	dh = int(oh*jitter)
	# pleft, pright, ptop, pbot
	# if negative: padding origin picture with (0, 0, 0),
	# if positive: crop origin picture.
	pleft = random.randint(-dw, dw)
	pright = random.randint(-dw, dw)
	ptop = random.randint(-dh, dh)
	pbot = random.randint(-dh, dh)
	swidth = ow - pleft - pright
	sheight = oh - ptop - pbot
	# sx: changed_ow / ow.
	# sy: changed_oh / oh.
	sx = float(swidth) / ow
	sy = float(sheight) / oh
	flip = random.randint(1, 10000) % 2
	cropped = img.crop((pleft, ptop, pleft+swidth-1, ptop+sheight-1))
	# dx, dy: the offset scale.
	# dx = pleft / changed_ow.
	# dy = ptop / changed_oh.
	dx = (float(pleft) / ow) / sx
	dy = (float(ptop) / oh) / sy
	# resize don't change the relative coord.
	sized = cropped.resize(shape)
	if flip:
		sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
	img = random_distort_image(sized, hue, saturation, exposure)
	return img, flip, dx, dy, sx, sy


# Get Label
def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
	max_boxes = 50
	label = np.zeros((max_boxes, 5))
	if os.path.getsize(labpath):
		bs = np.loadtxt(labpath)
		if bs is None:
			return label
		bs = np.reshape(bs, (-1, 5))
		cc = 0
		for i in range(bs.shape[0]):
			x1 = bs[i][1] - bs[i][3]/2
			y1 = bs[i][2] - bs[i][4]/2
			x2 = bs[i][1] + bs[i][3]/2
			y2 = bs[i][2] + bs[i][4]/2
			x1 = min(0.999, max(0, x1 * sx - dx))
			y1 = min(0.999, max(0, y1 * sy - dy))
			x2 = min(0.999, max(0, x2 * sx - dx))
			y2 = min(0.999, max(0, y2 * sy - dy))
			bs[i][1] = (x1 + x2) / 2
			bs[i][2] = (y1 + y2) / 2
			bs[i][3] = (x2 - x1)
			bs[i][4] = (y2 - y1)
			if flip:
				bs[i][1] = 0.999 - bs[i][1]
			# object is removed after data_augmentation.
			if bs[i][3] < 0.001 or bs[i][4] < 0.001:
				continue
			label[cc] = bs[i]
			cc += 1
	label = np.reshape(label, (-1))
	return label		


# Some Params Explain
# 	(1) saturation[for HSV saturation]: 
# 		Generate random sat_scale.
# 		sat_scale random choose from [1, saturation) and return sat_scale or 1.0 / sat_scale as sat_scale.
# 		sat = sat * sat_scale.
# 	(2) hue[for HSV hue]:
# 		Generate random add_hue.
# 		add_hue random choose from [-hue, hue), and product 255.
# 		hue = hue + add_hue.
# 	(3) exposure[for HSV value]:
# 		Generate random v_scale.
# 		v_scale random choose from [1, exposure) and return v_scale or 1.0 / v_scale as v_scale.
# 		v_scale = value * v_scale.
def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
	labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
	img = Image.open(imgpath).convert('RGB')
	img, flip, dx, dy, sx, sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
	label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
	return img, label