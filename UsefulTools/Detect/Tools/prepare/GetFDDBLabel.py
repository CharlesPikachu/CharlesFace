# Author: Charles
# Function:
# 	Transfer FDDB dataset labels.
# 	Just need class, x, y, w, h,
# 	Here we let class = 0 because of only delecting face.
import os
import numpy as np
import cv2


f = open('FDDB-fold-10-ellipseList.txt')
ori_img_path = 'E:\GraduationProject\Datasets\FDDB'
save_img_path = 'E:\images'
save_lab_path = 'E:\labels'
if not os.path.exists(save_img_path):
	os.mkdir(save_img_path)
if not os.path.exists(save_lab_path):
	os.mkdir(save_lab_path)


def label(f, num):
	gts = []
	for i in range(num):
		l = f.readline()
		l = l.strip()
		data = l.split(' ')
		major_axis_radius = float(data[0])
		minor_axis_radius = float(data[1])
		angle = float(data[2])
		c_x = float(data[3])
		c_y = float(data[4])
		class_ = 0
		gts.append([class_, major_axis_radius, minor_axis_radius, angle, c_x, c_y])
	return gts


start = 15396 + 290 + 285 + 274 + 302 + 298 + 302 + 279 + 276 + 259
img_num = start
while True:
	content = f.readline()
	temp = content.strip()
	img_path = temp + '.jpg'
	if not temp:
		break
	people_num = int(f.readline().strip())
	gts = label(f, people_num)
	if people_num > 50:
		continue
	lab_path_transfer = os.path.join(save_lab_path, str(img_num).zfill(6) + '.txt')
	img_path_transfer = os.path.join(save_img_path, str(img_num).zfill(6) + '.jpg')
	f_lab = open(lab_path_transfer, 'w')
	img = cv2.imread(os.path.join(ori_img_path, img_path))
	width = img.shape[1]
	height = img.shape[0]
	cv2.imwrite(img_path_transfer, img)
	for gt in gts:
		class_ = gt[0]
		major_axis_radius = gt[1]
		minor_axis_radius = gt[2]
		angle = gt[3] / 3.1415926 * 180
		center_x = gt[4]
		center_y = gt[5]
		mask = np.zeros((height, width), dtype=np.uint8)
		cv2.ellipse(mask, ((int)(center_x), (int)(center_y)), ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360.,(255, 255, 255))
		contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		r = cv2.boundingRect(contours[0])
		# x_min = r[0]
		# y_min = r[1]
		# x_max = r[0] + r[2]
		# y_max = r[1] + r[3]
		c_x = (r[0] + r[2]/2) / width
		c_y = (r[1] + r[3]/2) / height
		w = r[2] / width
		h = r[3] / height
		save_str = str(class_) + ' ' + str(c_x) + ' ' + str(c_y) + ' ' + str(w) + ' ' + str(h) + '\n'
		f_lab.write(save_str)
	f_lab.close()
	img_num += 1


f.close()
print('[INFO]: Get %d pictures...' % (img_num - start))