import json

with open('./kmeans_op.json', 'r') as f:
	a = json.load(f)
	print(type(a['num_anchors']))
	print(a['ann_dir'])
	if not a['img_dir']:
		print(2)
	f.close()