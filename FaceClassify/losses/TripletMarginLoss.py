# Author:
# 	Charles
# Function:
# 	Triplet loss function.
import torch
from torch.autograd import Function
import sys
sys.path.append('../')
from utils.utils import *


class TripletMarginLoss(Function):
	def __init__(self, margin):
		super(TripletMarginLoss, self).__init__()
		self.margin = margin
		# norm 2
		self.pdist = PairwiseDistance(2)
	def forward(self, anchor, positive, negative):
		dis_apos = self.pdist.forward(anchor, positive)
		dis_aneg = self.pdist.forward(anchor, negative)
		dist_hinge = torch.clamp(self.margin+dis_apos-dis_aneg, min=0.0)
		loss = torch.mean(dist_hinge)
		return loss