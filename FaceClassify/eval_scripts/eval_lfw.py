# Author:
# 	Charles
# Function:
# 	Test trained model in LFW.
import torch
import json
import numpy as np
import torchvision.transforms as transforms
from nets.ResNets import ResNet
from utils.eval import evaluate
from utils.utils import *
from torch.autograd import Variable
from datasets.LFWDataset import LFWDataset


f = open('config.json', 'r')
options = json.load(f)
f.close()
n_gpu = len(options['gpu'].split(','))


def test(test_loader, model, epoch):
	l2_distance = PairwiseDistance(2)
	model.eval()
	labels, distances = [], []
	for batch_idx, (data_a, data_b, label) in enumerate(test_loader):
		if use_cuda:
			data_a, data_b = data_a.cuda(), data_b.cuda()
		data_a, data_b, label = Variable(data_a, volatile=True), Variable(data_b, volatile=True), Variable(label)
		out_a, out_b = model(data_a), model(data_b)
		distance = l2_distance.forward(out_a, out_b)
		distances.append(distance.data.cpu().numpy())
		labels.append(label.data.cpu().numpy())
	labels = np.array([sublabel for label in labels for sublabel in label])
	distances = np.array([subdist for dist in distances for subdist in dist])
	TPR, FPR, accuracy, val, val_std, far = evaluate(distances, labels)
	logging('[Test Accuracy]: %f' % np.mean(accuracy))
	plot_roc(FPR, TPR, figure_name='roc_train_epoch_{}.png'.format(epoch), savefile='./log')


if __name__ == '__main__':
	use_cuda = True
	transform = transforms.Compose([
								Resize(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = [0.5, 0.5, 0.5],
													 std = [0.5, 0.5, 0.5])
							])
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
	test_imgs = LFWDataset(dir_='/home/zcjin/lfw/lfw_182', pairs_path='./lfw_pairs.txt', transform=transform)
	test_loader = torch.utils.data.DataLoader(test_imgs,
											  batch_size = 64,
											  shuffle = False,
											  **kwargs)
	model = ResNet(num_classes=10575, embeddings_num=128, img_size=224, is_fc=True, is_AvgPool=False, pretrained=False)
	if n_gpu > 1:
		model = torch.nn.DataParallel(model).cuda()
	else:
		model = model.cuda()
	resume = './weights/checkpoint_23439.pth'
	print('[INFO]: loading checkpoint {}'.format(resume))
	checkpoint = torch.load(resume)
	model.load_state_dict(checkpoint['state_dict'])
	epoch = checkpoint['epoch']
	test(test_loader, model, epoch)