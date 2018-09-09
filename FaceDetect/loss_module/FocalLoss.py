# paper: 
# 	yolo1: https://arxiv.org/abs/1506.02640
# 	yolo2: https://arxiv.org/abs/1612.08242
# 	yolo3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# Author: Charles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# This criterion is a implemenation of Focal Loss, 
# which is proposed in Focal Loss for Dense Object Detection.
# Input:
# 	alpha(1D Tensor, Variable): the scalar factor for this criterion.
# 	gamma(float, double): gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
# 						  putting more focus on hard, misclassiﬁed examples.
# 	size_average(bool): By default, the losses are averaged over observations for each minibatch.
# 						However, if the field size_average is set to False, the losses are
# 						instead summed for each minibatch.
# Function:
# 	Loss(x, class) -> 
class FocalLoss(nn.Module):
	def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
		super(FocalLoss, self).__init__()
		if alpha is None:
			self.alpha = Variable(torch.ones(class_num, 1))
		else:
			if isinstance(alpha, Variable):
				self.alpha = alpha
			else:
				self.alpha = Variable(alpha)
		self.gamma = gamma
		self.class_num = class_num
		self.size_average = size_average
	def forward(self, inputs, targets):
		N = inputs.size(0)
		C = inputs.size(1)
		# (N, C) Normalized
		P = F.softmax(inputs)
		# (N, C) all zero
		class_mask = inputs.data.new(N, C).fill_(0)
		class_mask = Variable(class_mask)
		# (N, 1)
		ids = targets.view(-1, 1)
		# class_mask[i][ids[i][j]] = 1
		# make each N' target class value = 1.
		class_mask.scatter_(1, ids.data, 1.)
		# all move to cuda
		if inputs.is_cuda and not self.alpha.is_cuda:
			self.alpha = self.alpha.cuda()
		# (N, 1)
		alpha = self.alpha[ids.data.view(-1).long()]
		# (N, C) -> (N) -> (N, 1)
		# each N' ground truth class possibility
		probs = (P*class_mask).sum(1).view(-1,1)
		# ln(each possibility)
		log_p = probs.log()
		batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
		if self.size_average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()
		return loss