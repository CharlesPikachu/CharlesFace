# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import time
from utils.standard_utils import *
from nets.darknet import Darknet
from PIL import Image


# detect one picture in PIL format.
def detect(cfgfile, weightfile, imgfile, num_classes, clsnamesfile=None, use_cuda=True):
	m = Darknet(cfgfile)
	m.print_network()
	m.is_yolov3 = True
	m.load_weights(weightfile)
	print('[INFO]:Loading weights from %s... Done!' % (weightfile))
	if not clsnamesfile:
		if num_classes == 20:
			namesfile = './data/voc.names'
		elif num_classes == 80:
			namesfile = './data/coco.names'
		else:
			print('[INFO]:Default choose fail, You should give the class names file...')
			exit(-1)
	else:
		namesfile = '%s' % clsnamesfile
	if use_cuda:
		m.cuda()
	img = Image.open(imgfile).convert('RGB')
	sized = img.resize((m.width, m.height))
	start = time.time()
	boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
	finish = time.time()
	print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
	class_names = load_class_names(namesfile)
	plot_boxes(img, boxes, 'predictions.jpg', class_names)


# cv2 image format detect.
def detect_cv2(cfgfile, weightfile, imgfile, clsnamesfile=None, use_cuda=True):
	import cv2
	m = Darknet(cfgfile)
	m.print_network()
	m.is_yolov3 = True
	m.load_weights(weightfile)
	print('[INFO]:Loading weights from %s... Done!' % (weightfile))
	if not clsnamesfile:
		if m.num_classes == 20:
			namesfile = 'data/voc.names'
		elif m.num_classes == 80:
			namesfile = 'data/coco.names'
		else:
			print('[INFO]:Default choose fail, You should give the class names file...')
			return None
	else:
		namesfile = '%s' % clsnamesfile
	if use_cuda:
		m.cuda()
	img = cv2.imread(imgfile)
	sized = cv2.resize(img, (m.width, m.height))
	sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
	start = time.time()
	boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
	finish = time.time()
	print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
	class_names = load_class_names(namesfile)
	plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


# scikit-image format detect.
def detect_skimage(cfgfile, weightfile, imgfile, clsnamesfile=None, use_cuda=True):
	# pip3 install scikit-image
	from skimage import io
	from skimage.transform import resize
	m = Darknet(cfgfile)
	m.print_network()
	m.is_yolov3 = True
	m.load_weights(weightfile)
	print('[INFO]:Loading weights from %s... Done!' % (weightfile))
	if not clsnamesfile:
		if m.num_classes == 20:
			namesfile = 'data/voc.names'
		elif m.num_classes == 80:
			namesfile = 'data/coco.names'
		else:
			print('[INFO]:Default choose fail, You should give the class names file...')
			return None
	else:
		namesfile = '%s' % clsnamesfile
	if use_cuda:
		m.cuda()
	img = io.imread(imgfile)
	sized = resize(img, (m.width, m.height)) * 255
	start = time.time()
	boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
	finish = time.time()
	print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
	class_names = load_class_names(namesfile)
	plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':
	detect('cfg/me/celeba_face.cfg', './weights/darknet53_celeba.weights', 'test_pic/test1.jpg', 1, clsnamesfile='./data/celeba_face.names')
	# detect_cv2('cfg/me/celeba_face.cfg', './weights/000006.weights', 'test_pic/1.jpg', clsnamesfile='./data/celeba_face.names', use_cuda=True)
	# detect_skimage('cfg/yolov3.cfg', './weights/yolov3.weights', 'test_pic/sciences.jpg', clsnamesfile=None, use_cuda=True)