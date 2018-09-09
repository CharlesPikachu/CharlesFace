# Function:
# 	Transfer wider_face dataset labels.
# 	Just need class, x, y, w, h
# 	Here we let class = 0 because of only delecting face.
# Author:
# 	Charles
# Order:
# 	Train -> Test -> validition
import os
from PIL import Image


'''
f = open('wider_face_train_bbx_gt.txt', 'r')
contents = f.readlines()
i = 0
j = 0
for c in contents:
	c = c.strip()
	try:
		int(c)
		if int(c) > 50:
			i += 1
	except:
		try:
			int(c.split()[0])
		except:
			j += 1
print(j-i)
'''


root_path = 'E:\GraduationProject\数据集\WiderFace\WIDER_train\images'
pic_save_path = 'E:\images'
lab_save_path = 'E:\labels'
if not os.path.exists(pic_save_path):
	os.mkdir(pic_save_path)
if not os.path.exists(lab_save_path):
	os.mkdir(lab_save_path)
origin_file = 'wider_face_train_bbx_gt.txt'
train_file_list = 'train_wider.txt'
test_file_list = 'test_wider.txt'


f1 = open(origin_file, 'r')
f2 = open(train_file_list, 'a')
# f2 = open(test_file_list, 'a')
content = f1.readline()
start = 0
img_num = start - 1
gts = []
content = content.strip()
img_path = os.path.join(root_path, content)
while content:
	try:
		gt_num = int(content)
	except:
		try:
			data = content.split(' ')
			int(data[0])
			assert len(data) == 10
			x = float(data[0])
			y = float(data[1])
			w = float(data[2])
			h = float(data[3])
			class_ = 0
			gts.append([class_, w, h, x, y])
		except:
			if img_num > start-1:
				assert len(gts) == gt_num
				if gt_num > 50:
					gts = []
				else:
					img = Image.open(img_path)
					width = img.width
					height = img.height
					filenamepic = str(img_num).zfill(6) + '.jpg'
					filenamelab = str(img_num).zfill(6) + '.txt'
					lab_f = open(os.path.join(lab_save_path, filenamelab), 'w')
					for gt in gts:
						class_ = gt[0]
						x = (gt[3]+gt[1]/2) / float(width)
						y = (gt[4]+gt[2]/2) / float(height)
						w = gt[1] / float(width)
						h = gt[2] / float(height)
						save_str = str(class_) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
						lab_f.write(save_str)
					img.save(os.path.join(pic_save_path, filenamepic))
					lab_f.close()
					img_num += 1
					gts = []
				img_path = os.path.join(root_path, content)
			else:
				img_num += 1
	content = f1.readline()
	content = content.strip()


assert len(gts) == gt_num
if gt_num > 50:
	gts = []
else:
	img = Image.open(img_path)
	width = img.width
	height = img.height
	filenamepic = str(img_num).zfill(6) + '.jpg'
	filenamelab = str(img_num).zfill(6) + '.txt'
	lab_f = open(os.path.join(lab_save_path, filenamelab), 'w')
	for gt in gts:
		class_ = 0
		x = (gt[3]+gt[1]/2) / float(width)
		y = (gt[4]+gt[2]/2) / float(height)
		w = gt[1] / float(width)
		h = gt[2] / float(height)
		save_str = str(class_) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
		lab_f.write(save_str)
	img.save(os.path.join(pic_save_path, filenamepic))
	lab_f.close()
	img_num += 1


print('[INFO]: Get %d pictures for train.' % (img_num-start))
f1.close()
f2.close()