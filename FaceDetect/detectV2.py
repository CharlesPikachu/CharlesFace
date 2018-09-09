# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
from utils.standard_utils import *
from nets.darknet import Darknet
from PIL import Image
import torch


# only for one picture
def detect(cfgfile, weightfile, imgfile, clsnamesfile=None, use_cuda=True):
	m = Darknet(cfgfile)
	m.print_network()
	m.load_weights(weightfile)
	print('[INFO]:Loading weights from %s... Done!' % (weightfile))
	if not clsnamesfile:
		if m.num_classes == 20:
			namesfile = 'data/voc.names'
		elif m.num_classes == 80:
			namesfile = 'data/coco.names'
		else:
			print('[INFO]:Default choose fail, You should give the class names file...')
			exit(-1)
	else:
		namesfile = '%s' % clsnamesfile
	if use_cuda:
		m.cuda()
	img = Image.open(imgfile).convert('RGB')
	sized = img.resize((m.width, m.height))
	conf_thresh = 0.2
	nms_thresh = 0.3
	m.eval()
	if isinstance(sized, Image.Image):
		sized = image2torch(sized)
	elif type(sized) == np.ndarray:
		sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
	else:
		print("[Error]: unknow image type...")
		exit(-1)
	if use_cuda:
		sized = sized.cuda()
	sized = torch.autograd.Variable(sized)
	output = m(sized)
	output = output.data
	# by default, we only get one picture detected boxes.
	anchors = [anchor/m.loss.stride for anchor in m.anchors]
	boxes = get_region_boxes(output, conf_thresh, m.num_classes, anchors, m.num_anchors)[0]
	boxes = nms(boxes, nms_thresh)
	class_names = load_class_names(namesfile)
	plot_boxes(img, boxes, 'predictions.jpg', class_names)



if __name__ == '__main__':
	# , clsnamesfile='data/teachers.names'
	img = './frames/%s.jpg' % 2886
	detect('./cfg/me/darknet19_wfc_face.cfg', './weights/darknet19_wfc.weights', img, clsnamesfile='data/face.names')