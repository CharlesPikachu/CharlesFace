# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
from utils.standard_utils import *
from nets.darknet import Darknet
import cv2


def Demo(cfgfile, weightfile, clsnamesfile=None, use_cuda=True):
	m = Darknet(cfgfile)
	# m.print_net()
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
	class_names = load_class_names(namesfile)
	if use_cuda:
		m.cuda()
	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		print('[Error]:Unable to open camera...')
		exit(-1)
	while True:
		res, img = capture.read()
		if res:
			sized = cv2.resize(img, (m.width, m.height))
			bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
			draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
			cv2.imshow(cfgfile, draw_img)
			cv2.waitKey(1)
			import time
			time.sleep(0.2)
		else:
			print('[Error]:Unable to read image...')
			exit(-1)



if __name__ == '__main__':
	Demo('cfg/me/celeba_face.cfg', './weights/000105.weights', clsnamesfile='./data/celeba_face.names')
	# Demo('cfg/yolov3.cfg', './weights/yolov3.weights', clsnamesfile='./data/coco.names')