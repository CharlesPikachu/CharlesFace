# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
from utils.standard_utils import *
from utils.cfg_utils import *
import time


# Training settings
class Settings():
	# datacfg = './cfg/celeba_face.data'
	# datacfg = './cfg/wider_fddb_celeba_face.data'
	datacfg = './cfg/darknet19_wfc_face.data'
	# cfgfile = './cfg/celeba_face.cfg'
	# cfgfile = './cfg/wider_fddb_celeba_face.cfg'
	cfgfile = './cfg/darknet19_wfc_face.cfg'
	weightfile = './weights/%s.weights' % str(time.strftime("<%Y-%m-%d>%H:%M:%S", time.localtime()))
	# weightfile = './backup/000087.weights'
	# weightfile = './weights/yolov3.weights'
	data_options = read_data_cfg(datacfg)
	net_options = parse_cfg(cfgfile)[0]
	trainlist = data_options['train']
	testlist = data_options['valid']
	# backup
	backupdir = data_options['backup']
	nsamples = file_lines(trainlist)
	gpus = data_options['gpus']
	ngpus = len(gpus.split(','))
	num_workers = int(data_options['num_workers'])
	batch_size = int(net_options['batch'])
	max_batches = int(net_options['max_batches'])
	learning_rate = float(net_options['learning_rate'])
	momentum = float(net_options['momentum'])
	# used in weight decay (L2 penalty)
	decay = float(net_options['decay'])
	steps = [float(step) for step in net_options['steps'].split(',')]
	scales = [float(scale) for scale in net_options['scales'].split(',')]
	# whether YOLOv3 model or not.
	YOLOv3 = True
	# whether multiples train or not.
	is_multi_train = False
	multi_img_shapes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
	# HParams Train
	max_epochs = max_batches*batch_size // nsamples + 1
	use_cuda = True
	seed = int(time.time())
	# avoid Denominator is zero
	eps = 1e-5
	# epoches
	save_interval = 10
	# batches
	dot_interval = 70
	# HParams Test
	conf_thresh = 0.25
	nms_thresh = 0.4
	iou_thresh = 0.5



if __name__ == '__main__':
	print(Settings().batch_size)