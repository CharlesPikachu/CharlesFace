# Draw network structure
from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot
from nets.darknet import Darknet


model = Darknet('./cfg/me/darknet19_wfc_face.cfg')
model.print_network()
x = Variable(torch.randn(1, 3, 416, 416), requires_grad=True)
y = model(x)
params_dict = dict(model.named_parameters())
params_dict['x'] = x
g = make_dot(y, params=params_dict)
g.view()
