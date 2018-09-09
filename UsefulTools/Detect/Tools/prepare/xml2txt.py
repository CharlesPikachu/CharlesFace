# Author: Charles
# xml label -> txt label
import os
import xml.etree.ElementTree as ET
classes = []


def convert(size, box):
	dw = 1. / size[0]
	dh = 1. / size[1]
	x = (box[0] + box[1]) / 2.0
	y = (box[2] + box[3]) / 2.0
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return (x, y, w, h)


# flag:
# 	whether give class names or not.
def parse_lxml(lxml_path, obj_path=None, flag=False, saveclasses=None):
	if obj_path is None:
		obj_path = lxml_path.replace('Lxml', 'labels')
	in_file = open(lxml_path)
	out_file = open(obj_path, 'w')
	tree = ET.parse(in_file)
	root = tree.getroot()
	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)
	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		cls_ = obj.find('name').text
		if flag:
			if cls_ not in classes or int(difficult) == 1:
				continue
			cls_id = classes.index(cls_)
		else:
			if int(difficult) == 1:
				continue
			if cls_ not in classes:
				classes.append(cls_)
			cls_id = classes.index(cls_)
		xmlbox = obj.find('bndbox')
		tmp = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
		xywh = convert((w, h), tmp)
		out_file.write(str(cls_id) + " " + " ".join([str(a) for a in xywh]) + '\n')
	in_file.close()
	out_file.close()
	if saveclasses is not None:
		with open(saveclasses, 'w') as f:
			for cls_ in classes:
				f.write(cls_ + '\n')
		f.close()




if __name__ == '__main__':
	IMAGE_NUM = 79
	lxml_dir = os.getcwd() + '/Lxml/{}'
	obj_dir = os.getcwd() + '/labels/{}'
	if not os.path.exists(obj_dir[0: -3]):
		os.makedirs(obj_dir[0: -3])
	for i in range(IMAGE_NUM):
		if len(str(i)) == 1:
			lxml_path = lxml_dir.format('00000%s.xml' % str(i))
			obj_path = obj_dir.format('00000%s.txt' % str(i))
		elif len(str(i)) == 2:
			lxml_path = lxml_dir.format('0000%s.xml' % str(i))
			obj_path = obj_dir.format('0000%s.txt' % str(i))
		elif len(str(i)) == 3:
			lxml_path = lxml_dir.format('000%s.xml' % str(i))
			obj_path = obj_dir.format('000%s.txt' % str(i))
		elif len(str(i)) == 4:
			lxml_path = lxml_dir.format('00%s.xml' % str(i))
			obj_path = obj_dir.format('00%s.txt' % str(i))
		elif len(str(i)) == 5:
			lxml_path = lxml_dir.format('0%s.xml' % str(i))
			obj_path = obj_dir.format('0%s.txt' % str(i))
		elif len(str(i)) == 5:
			lxml_path = lxml_dir.format('%s.xml' % str(i))
			obj_path = obj_dir.format('%s.txt' % str(i))
		else:
			print('[Error]:Pic num too large...')
			break
		parse_lxml(lxml_path, obj_path, saveclasses='teachers.names')