# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import os
import torch
import dataset
from torch.autograd import Variable
from torchvision import transforms
from utils.standard_utils import *
from nets.darknet import Darknet


def valid(datacfg, cfgfile, weightfile, outfile, use_cuda=True):
	options = read_data_cfg(datacfg)
	valid_images = options['valid']
	name_list = options['names']
	prefix = 'results'
	names = load_class_names(name_list)
	with open(valid_images) as fp:
		tmp_files = fp.readlines()
		valid_files = [item.rstrip() for item in tmp_files]
	m = Darknet(cfgfile)
	m.print_network()
	m.load_weights(weightfile)
	if use_cuda:
		m.cuda()
	m.eval()
	valid_dataset = dataset.listDataset(valid_images, 
										shape=(m.width, m.height),
										shuffle=False,
										transform=transforms.Compose([transforms.ToTensor(),]))
	valid_batchsize = 2
	assert(valid_batchsize > 1)
	kwargs = {'num_workers': 4, 'pin_memory': True}
	valid_loader = torch.utils.data.DataLoader(valid_dataset, 
											   batch_size=valid_batchsize, 
											   shuffle=False, 
											   **kwargs)
	fps = [0] * m.num_classes
	if not os.path.exists(prefix):
		os.mkdir(prefix)
	for i in range(m.num_classes):
		buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
		fps[i] = open(buf, 'w')
	lineId = -1
	conf_thresh = 0.005
	nms_thresh = 0.45
	for batch_idx, (data, target) in enumerate(valid_loader):
		if use_cuda:
			data = data.cuda()
		data = Variable(data, volatile=True)
		output = m(data).data
		batch_boxes = get_region_boxes(output, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)
		for i in range(output.size(0)):
			lineId = lineId + 1
			fileId = os.path.basename(valid_files[lineId]).split('.')[0]
			width, height = get_image_size(valid_files[lineId])
			print(valid_files[lineId])
			boxes = batch_boxes[i]
			boxes = nms(boxes, nms_thresh)
			for box in boxes:
				x1 = (box[0] - box[2]/2.0) * width
				y1 = (box[1] - box[3]/2.0) * height
				x2 = (box[0] + box[2]/2.0) * width
				y2 = (box[1] + box[3]/2.0) * height
				det_conf = box[4]
				for j in range((len(box)-5) // 2):
					cls_conf = box[5+2*j]
					cls_id = box[6+2*j]
					prob = det_conf * cls_conf
					fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))
	for i in range(m.num_classes):
		fps[i].close()