# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
# Function:
# 	kmeans get anchors for training
import random
import os
import numpy as np
import json
import xml.etree.ElementTree as ET
import pickle
from pyecharts import Line


# parse xml files
def parse_annotation(ann_dir, img_dir, labels=[], save_name='annotation.pkl', count_max=None):
	# print('[INFO]:parse_annotation fun running...')
	all_instances = []
	seen_labels = {}
	if count_max:
		i = 0
	for ann in sorted(os.listdir(ann_dir)):
		img = {'object': []}
		tree = ET.parse(ann_dir + ann)
		for elem in tree.iter():
			if 'filename' in elem.tag:
				img['filename'] = img_dir + elem.text
			elif 'width' in elem.tag:
				img['width'] = int(elem.text)
			elif 'height' in elem.tag:
				img['height'] = int(elem.text)
			elif 'object' in elem.tag:
				obj = {}
				for attr in list(elem):
					if 'name' in attr.tag:
						obj['name'] = attr.text
						if obj['name'] in seen_labels:
							seen_labels[obj['name']] += 1
						else:
							seen_labels[obj['name']] = 1
						if len(labels) > 0 and obj['name'] not in labels:
							break
						else:
							img['object'] += [obj]
					if 'bndbox' in attr.tag:
						for dim in list(attr):
							if 'xmin' in dim.tag:
								obj['xmin'] = int(round(float(dim.text)))
							elif 'ymin' in dim.tag:
								obj['ymin'] = int(round(float(dim.text)))
							elif 'xmax' in dim.tag:
								obj['xmax'] = int(round(float(dim.text)))
							elif 'ymax' in dim.tag:
								obj['ymax'] = int(round(float(dim.text)))
		if len(img['object']) > 0:
			all_instances += [img]
		if count_max:
			i += 1
			if i > count_max-1:
				break
	# temp = {'all_instances': all_instances, 'seen_labels': seen_labels}
	# with open(save_name, 'wb') as f:
	# 	pickle.dump(temp, f, protocol=pickle.HIGHEST_PROTOCOL)
	# f.close()
	# print(len(all_instances))
	return all_instances, seen_labels


# save anchors
def save_anchors(save_anchs):
	from openpyxl import Workbook
	wb = Workbook()
	ws = wb.active
	ws.append(['width', 'height'])
	for sa in save_anchs:
		try:
			ws.append([sa[0], sa[1]])
		except:
			print('[WARNING]:An anchor lost...')
			continue
	if not os.path.exists('./results'):
		os.mkdir('./results')
	wb.save('./results/' + 'Anchors.xlsx')
	print('[INFO]:Anchors saved to excel successfully...')


# print_anchors
def print_anchors(centeranchors, width, height, save=True):
	outstring = '[Anchors]: \n'
	anchors = centeranchors.copy()
	widths = anchors[:, 0]
	sorted_inds = np.argsort(widths)
	if save:
		save_anchs = []
	for i in sorted_inds:
		if save:
			save_anchs.append([int(round(anchors[i, 0] * width)), int(round(anchors[i, 1] * height))])
		outstring += str(int(round(anchors[i, 0] * width))) + ', ' + str(int(round(anchors[i, 1] * height))) + '\n'
	if save:
		save_anchors(save_anchs)
	print(outstring[0: -1])


# compute IOU
def IOU(wh, centeranchors):
	# print('[INFO]:IOU fun running...')
	w, h = wh
	results = []
	for ca in centeranchors:
		c_w, c_h = ca
		if c_w >= w and c_h >= h:
			result = w * h / (c_w * c_h)
		elif c_w >= w and c_h <= h:
			result = w * c_h / (w * h + c_w * c_h - w * c_h)
		elif c_w <= w and c_h >= h:
			result = c_w * h / (w * h + c_w * c_h - c_w * h)
		else:
			result = c_w * c_h / (w * h)
		results.append(result)
	return np.array(results)


# compute average IOU
def avgIOU(whs, centeranchors):
	# print('[INFO]:avgIOU fun running...')
	ann_num, anchor_dim = whs.shape
	sum_ = 0
	for i in range(ann_num):
		# min 1 - IOU is equal to max IOU
		sum_ += max(IOU(whs[i], centeranchors))
	return sum_ / ann_num


# draw kmeans distance trends.
def DrawDistance(distances):
	line = Line('聚类距离走势图')
	attrs = []
	values = []
	i = 1
	for d in distances:
		attrs.append(i)
		values.append(d)
		i += 1
	line.add("距离", attrs, values, is_smooth=False, mark_point=["average", "max", 'min'])
	line.render('kmeansLine.html')


# kmeans
def kmeans(whs, num_anchors):
	# print('[INFO]:kmeans fun running...')
	# annotation num
	ann_num, anchor_dim = whs.shape
	iteration = 0
	# size: (ann_num)
	prev_assignments = np.ones(ann_num) * (-1)
	prev_distances = np.zeros((ann_num, num_anchors))
	# elements in [0, ann_num-1], allow repeat.
	# inds size: (num_anchors)
	inds = [random.randrange(ann_num) for i in range(num_anchors)]
	# kmeans center
	centeranchors = whs[inds]
	all_distances = []
	while True:
		distances = []
		iteration += 1
		for i in range(ann_num):
			d = 1 - IOU(whs[i], centeranchors)
			distances.append(d)
		# size: (ann_num, num_anchors)
		distances = np.array(distances)
		print('[Iteration {}]: distances = {}'.format(iteration, np.sum(np.abs(prev_distances - distances))))
		all_distances.append(np.sum(np.abs(prev_distances - distances)))
		# the min element index along axis 1.
		# size: (ann_num)
		assignments = np.argmin(distances, axis=1)
		# all elements equal
		if (assignments == prev_assignments).all():
			DrawDistance(all_distances)
			return centeranchors
		# calculate new centeranchors
		# (num_anchors, anchor_dim)
		centeranchors_sum = np.zeros((num_anchors, anchor_dim), np.float)
		for i in range(ann_num):
			centeranchors_sum[assignments[i]] += whs[i]
		for j in range(num_anchors):
			centeranchors[j] = centeranchors_sum[j] / (np.sum(assignments == j) + 1e-6)
		prev_assignments = assignments.copy()
		prev_distances = distances.copy()


# '''
# call the run function to get w,h kmeans.
def run():
	f = open('./kmeans_op.json', 'r')
	options = json.load(f)
	f.close()
	num_anchors = int(options['num_anchors'])
	width = int(options['width'])
	height = int(options['height'])
	train_imgs, train_lables = parse_annotation(
												ann_dir = options['ann_dir'],
												img_dir = options['img_dir'] if options['img_dir'] else '',
												labels = options['labels'] if options['labels'] else [],
												save_name = options['save_name'] if options['save_name'] else 'annotation.pkl',
												count_max = options['count_max'] if options['count_max'] else None
												)
	allWH_relative = []
	for img in train_imgs:
		# print(img['filename'])
		for obj in img['object']:
			relative_w = (float(obj['xmax']) - float(obj['xmin'])) / img['width']
			relative_h = (float(obj['ymax']) - float(obj['ymin'])) / img['height']
			allWH_relative.append(tuple(map(float, (relative_w, relative_h))))
	allWH_relative = np.array(allWH_relative)
	centeranchors = kmeans(allWH_relative, num_anchors)
	print('[Num_Anchors]: %s, [Average IOU]: %.2f' % (num_anchors, avgIOU(allWH_relative, centeranchors)))
	print_anchors(centeranchors, width, height, save=True)
# '''


'''
def run():
	f = open('./kmeans_op.json', 'r')
	options = json.load(f)
	f.close()
	num_anchors = int(options['num_anchors'])
	width = int(options['width'])
	height = int(options['height'])
	txt_dir = options['txt_dir']
	allWH_relative = []
	all_labels = sorted(os.listdir(txt_dir))
	for label in all_labels:
		fl = open(os.path.join(txt_dir, label), 'r')
		gts = fl.readlines()
		for gt in gts:
			gt = gt.strip()
			if not gt:
				break
			gt = gt.split(' ')
			assert len(gt) == 5
			relative_w = float(gt[3])
			relative_h = float(gt[4])
			allWH_relative.append([relative_w, relative_h])
	allWH_relative = np.array(allWH_relative)
	centeranchors = kmeans(allWH_relative, num_anchors)
	print('[Num_Anchors]: %s, [Average IOU]: %.2f' % (num_anchors, avgIOU(allWH_relative, centeranchors)))
	print_anchors(centeranchors, width, height, save=True)
'''




if __name__ == '__main__':
	run()