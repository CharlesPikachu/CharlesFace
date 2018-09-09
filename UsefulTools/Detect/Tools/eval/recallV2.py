# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
from PIL import Image
from utils.standard_utils import *
from nets.darknet import Darknet


# compute recall
def eval_list(cfgfile, weightfile, imglist, use_cuda=True):
	m = Darknet(cfgfile)
	m.eval()
	m.load_weights(weightfile)
	eval_wid = m.width
	eval_hei = m.height
	if use_cuda:
		m.cuda()
	conf_thresh = 0.25
	nms_thresh = 0.4
	iou_thresh = 0.5
	min_box_scale = 8. / m.width
	with open(imglist) as fp:
		lines = fp.readlines()
	total = 0.0
	proposals = 0.0
	correct = 0.0
	lineId = 0
	avg_iou = 0.0
	for line in lines:
		img_path = line.rstrip()
		if img_path[0] == '#':
			continue
		lineId = lineId + 1
		lab_path = img_path.replace('images', 'labels')
		lab_path = lab_path.replace('JPEGImages', 'labels')
		lab_path = lab_path.replace('.jpg', '.txt').replace('.png', '.txt')
		truths = read_truths_args(lab_path, min_box_scale)
		img = Image.open(img_path).convert('RGB').resize((eval_wid, eval_hei))
		boxes = do_detect(m, img, conf_thresh, nms_thresh, use_cuda)
		if False:
			savename = "tmp/%06d.jpg" % (lineId)
			print("save %s" % savename)
			plot_boxes(img, boxes, savename)
		total = total + truths.shape[0]
		for i in range(len(boxes)):
			if boxes[i][4] > conf_thresh:
				proposals += 1
		for i in range(truths.shape[0]):
			box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0]
			best_iou = 0
			for j in range(len(boxes)):
				iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
				best_iou = max(iou, best_iou)
			if best_iou > iou_thresh:
				avg_iou += best_iou
				correct += 1
	precision = 1.0*correct / proposals
	recall = 1.0*correct / total
	fscore = 2.0*precision*recall / (precision+recall)
	print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" % (lineId-1, avg_iou/correct, recall, precision, fscore))