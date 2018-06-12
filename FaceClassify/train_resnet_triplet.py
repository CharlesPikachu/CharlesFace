# Author:
# 	Charles
# Function:
# 	Train a Face recognizer based on resnet.
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from utils.utils import *
from datasets.Dataset import *
from datasets.LFWDataset import LFWDataset
from torch.autograd import Variable
from nets.ResNets import ResNet


# Get config.json Hyperparameters
# ------------------------------------------------------------------------------------------------
f = open('config.json', 'r')
options = json.load(f)
f.close()
n_gpu = len(options['gpu'].split(','))
os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
use_cuda = True if torch.cuda.is_available() else False
num_workers = options['num_workers']
train_dir = options['train_dir']
test_dir = options['test_dir']
pairs_dir = options['pairs_dir']
log_dir = options['log_dir']
batch_size = int(options['batch_size'])
embedding_size = int(options['embedding_size'])
lr = float(options['lr'])
lr_decay = float(options['lr_decay'])
weight_decay = float(options['weight_decay'])
optimizer_choice = options['optimizer']
margin = float(options['margin'])
backup_dir = options['backup_dir']
resume = options['resume']
max_batches = int(options['max_batches'])
batch_interval = int(options['batch_interval'])
img_size = int(options['img_size'])
triplet_weight = float(options['triplet_weight'])
cross_entropy_weight = float(options['cross_entropy_weight'])
pretrained = bool(options['pretrained'])
n_triplets = int(options['n_triplets'])
pre_gen = bool(options['pre_gen'])
if pretrained and resume:
	print('[Error]: pretrained=True and resume=True is unpermitted...')
	exit(-1)
# ------------------------------------------------------------------------------------------------


# pre-process
# ------------------------------------------------------------------------------------------------
if not os.path.exists(backup_dir):
	os.mkdir(backup_dir)
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
l2_distance = PairwiseDistance(2)
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
if use_cuda:
	cudnn.benchmark = True
transform = transforms.Compose([
								transforms.RandomResizedCrop(size=img_size),
								transforms.RandomHorizontalFlip(),
								transforms.ColorJitter(),
								transforms.ToTensor(),
								transforms.Normalize(mean = [0.5, 0.5, 0.5],
													 std = [0.5, 0.5, 0.5])
							])
if not pre_gen:
	train_imgs = LTripletImageDataset(train_dir, transform=transform, max_pairs=max_batches*batch_size)
else:
	if not resume:
		train_imgs = TripletFaceDataset(dir_=train_dir, n_triplets=n_triplets, transform=transform)
		save_triplets(train_imgs.training_triplets, save_path='%s/training_triplets.list' % backup_dir)
	else:
		train_imgs = TripletFaceDataset(dir_=train_dir, n_triplets=n_triplets, transform=transform)
		try:
			print('[INFO]: Read triplets from default folder...')
			train_imgs.training_triplets = read_triplets(save_path='%s/training_triplets.list' % backup_dir)
		except:
			print('[Warning]: Fail in reading, new triplets generate...')
			save_triplets(train_imgs.training_triplets, save_path='%s/training_triplets.list' % backup_dir)
train_loader = torch.utils.data.DataLoader(train_imgs,
										   batch_size = batch_size,
										   shuffle = True,
										   **kwargs)
transform_test = transforms.Compose([
								Resize(img_size),
								transforms.ToTensor(),
								transforms.Normalize(mean = [0.5, 0.5, 0.5],
													 std = [0.5, 0.5, 0.5])
							])
if pre_gen:
	batch_interval = len(train_loader)
test_imgs = LFWDataset(dir_=test_dir, pairs_path=pairs_dir, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_imgs,
										  batch_size = batch_size,
										  shuffle = False,
										  **kwargs)
# ------------------------------------------------------------------------------------------------


# Train and Test.
# ------------------------------------------------------------------------------------------------
def adjust_learning_rate(optimizer, step=None):
	for group in optimizer.param_groups:
		if 'step' not in group:
			if step:
				group['step'] = step
			else:
				group['step'] = 0
		group['step'] += 1
		group['lr'] = lr / (1 + group['step'] * lr_decay)
	return group['step'], group['lr']


def create_optimizer(model, lr):
	if optimizer_choice == 'sgd':
		optimizer = optim.SGD(model.parameters(), 
							  lr = lr,
							  momentum = 0.9,
							  dampening = 0.9,
							  weight_decay = weight_decay)
	elif optimizer_choice == 'adam':
		optimizer = optim.Adam(model.parameters(),
							   lr = lr,
							   weight_decay = weight_decay
							   )
	elif optimizer_choice == 'adagrad':
		optimizer = optim.Adagrad(model.parameters(),
								  lr = lr,
								  lr_decay = lr_decay,
								  weight_decay = weight_decay)
	else:
		print('[Error]:<create_optimizer in train.py> optimizer unsupported now...')
		exit(-1)
	return optimizer


def test(test_loader, model, batch_num):
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
	plot_roc(FPR, TPR, figure_name='roc_train_batch_{}.png'.format(batch_num), savefile=log_dir)


def train(train_loader, model, optimizer, start_batch, max_batches, step=0, epoch='non'):
	model.train()
	for batch_idx, (anchor, positive, negative, label_p, label_n) in enumerate(train_loader):
		batch_total = batch_idx + start_batch + 1
		if batch_total > max_batches:
			break
		if use_cuda:
			anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
		anchor, positive, negative = Variable(anchor, requires_grad=True), Variable(positive, requires_grad=True), Variable(negative, requires_grad=True)
		out_an, out_po, out_ne = model(anchor), model(positive), model(negative)
		dis_apos = l2_distance.forward(out_an, out_po)
		dis_aneg = l2_distance.forward(out_an, out_ne)
		all_compares = (dis_aneg - dis_apos < margin).cpu().data.numpy().flatten()
		# all_compares = torch.eq(dis_aneg - dis_apos < margin, dis_aneg - dis_apos > 0).cpu().data.numpy().flatten()
		hard_triplets = np.where(all_compares == 1)
		if len(hard_triplets[0]) == 0:
			continue
		if use_cuda:
			in_selected_a = Variable(torch.from_numpy(anchor.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			in_selected_p = Variable(torch.from_numpy(positive.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			in_selected_n = Variable(torch.from_numpy(negative.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			out_selected_a = Variable(torch.from_numpy(out_an.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			out_selected_p = Variable(torch.from_numpy(out_po.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			out_selected_n = Variable(torch.from_numpy(out_ne.cpu().data.numpy()[hard_triplets]).cuda(), requires_grad=True)
			# in_selected_a = Variable(torch.from_numpy(anchor.cpu().data.numpy()[hard_triplets]).cuda())
			# in_selected_p = Variable(torch.from_numpy(positive.cpu().data.numpy()[hard_triplets]).cuda())
			# in_selected_n = Variable(torch.from_numpy(negative.cpu().data.numpy()[hard_triplets]).cuda())
			# out_selected_a = Variable(torch.from_numpy(out_an.cpu().data.numpy()[hard_triplets]).cuda())
			# out_selected_p = Variable(torch.from_numpy(out_po.cpu().data.numpy()[hard_triplets]).cuda())
			# out_selected_n = Variable(torch.from_numpy(out_ne.cpu().data.numpy()[hard_triplets]).cuda())
		else:
			in_selected_a = Variable(torch.from_numpy(anchor.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			in_selected_p = Variable(torch.from_numpy(positive.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			in_selected_n = Variable(torch.from_numpy(negative.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			out_selected_a = Variable(torch.from_numpy(out_an.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			out_selected_p = Variable(torch.from_numpy(out_po.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			out_selected_n = Variable(torch.from_numpy(out_ne.cpu().data.numpy()[hard_triplets]), requires_grad=True)
			# in_selected_a = Variable(torch.from_numpy(anchor.cpu().data.numpy()[hard_triplets]))
			# in_selected_p = Variable(torch.from_numpy(positive.cpu().data.numpy()[hard_triplets]))
			# in_selected_n = Variable(torch.from_numpy(negative.cpu().data.numpy()[hard_triplets]))
			# out_selected_a = Variable(torch.from_numpy(out_an.cpu().data.numpy()[hard_triplets]))
			# out_selected_p = Variable(torch.from_numpy(out_po.cpu().data.numpy()[hard_triplets]))
			# out_selected_n = Variable(torch.from_numpy(out_ne.cpu().data.numpy()[hard_triplets]))
		selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
		selected_label_n = torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
		if use_cuda:
			selected_label_p = selected_label_p.cuda()
			selected_label_n = selected_label_n.cuda()
		triplet_criterion = nn.TripletMarginLoss(margin=margin)
		triplet_loss = triplet_criterion(out_selected_a, out_selected_p, out_selected_n) * triplet_weight
		if n_gpu > 1:
			cur_model = model.module
		else:
			cur_model = model
		CrossEntropycriterion = nn.CrossEntropyLoss()
		cls_anchor = cur_model.forward_classifier(in_selected_a)
		cls_postive = cur_model.forward_classifier(in_selected_p)
		cls_negative = cur_model.forward_classifier(in_selected_n)
		predicted_labels = torch.cat([cls_anchor, cls_postive, cls_negative])
		true_labels = torch.cat([Variable(selected_label_p), Variable(selected_label_p), Variable(selected_label_n)])
		cross_entropy_loss = CrossEntropycriterion(predicted_labels, true_labels) * cross_entropy_weight
		loss = cross_entropy_loss + triplet_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx == 0:
			step_now, lr_now = adjust_learning_rate(optimizer, step)
		else:
			step_now, lr_now = adjust_learning_rate(optimizer)
		logging('[Epoch-BATCH]: %s-%d, [LR]: %f, [step]: %d' % (str(epoch), batch_total, lr_now, step_now))
		msg = '[Loss]: <triplet_loss>: %f, <cross_entropy_loss>: %f, <total loss>: %f' % (triplet_loss, cross_entropy_loss, loss)
		logging(msg)
		if (batch_total % batch_interval == 0) or (batch_total == max_batches):
			torch.save({'step': step_now, 'batch_idx': batch_total, 'state_dict': model.state_dict()}, './{}/checkpoint_{}.pth'.format(backup_dir, batch_total))
			test(test_loader, model, batch_total)
			if batch_total == max_batches:
				break


def run():
	# resnet34 by default.
	if not pre_gen:
		model = ResNet(resnet='resnet50', num_classes=train_imgs.num_classes, embeddings_num=embedding_size, img_size=img_size, is_fc=True, is_AvgPool=False, pretrained=pretrained)
	else:
		model = ResNet(resnet='resnet50', num_classes=len(train_imgs.classes), embeddings_num=embedding_size, img_size=img_size, is_fc=True, is_AvgPool=False, pretrained=pretrained)
	if use_cuda:
		if n_gpu > 1:
			model = nn.DataParallel(model).cuda()
			cur_model = model.module
		else:
			model = model.cuda()
			cur_model = model
	if resume:
		if os.path.isfile(resume):
			print('[INFO]: loading checkpoint {}...'.format(resume))
			checkpoint = torch.load(resume)
			step = checkpoint['step']
			batch_idx = checkpoint['batch_idx']
			model.load_state_dict(checkpoint['state_dict'])
		else:
			print('[Warning]: cannot load checkpoint {}, start new train...'.format(resume))
			batch_idx = 0
			step = 0
	else:
		batch_idx = 0
		step = 0
	optimizer = create_optimizer(model, lr)
	if not pre_gen:
		train(train_loader, model, optimizer, batch_idx, max_batches, step=step)
	else:
		start_epoch = batch_idx // len(train_loader)
		end_epoch = max_batches // len(train_loader)
		for epoch in range(start_epoch, end_epoch):
			train(train_loader, model, optimizer, batch_idx, max_batches, step=step, epoch=epoch)
			batch_idx += len(train_loader)
# ------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	run()