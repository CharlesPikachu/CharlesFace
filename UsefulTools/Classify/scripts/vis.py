# Draw network structure
from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot
from nets.ResNets import ResNet


model = ResNet(resnet='resnet18', num_classes=10575, embeddings_num=128, img_size=224, is_fc=True, is_AvgPool=False)
x = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
y = model(x)
params_dict = dict(model.named_parameters())
params_dict['x'] = x
g = make_dot(y, params=params_dict)
g.view()
