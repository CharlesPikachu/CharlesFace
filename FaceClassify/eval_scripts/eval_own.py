# Author:
# 	Charles
# Function:
# 	Test a Face recognizer with Pytorch using my own pic.
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.utils import *
from nets.ResNets import ResNet
from torch.autograd import Variable


def test(img1_path, img2_path, model, use_cuda=True):
	l2_distance = PairwiseDistance(2)
	model.eval()
	img1 = Image.open(img1_path).convert('RGB')
	img2 = Image.open(img2_path).convert('RGB')
	transform = transforms.Compose([
								Resize(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = [0.5, 0.5, 0.5],
													 std = [0.5, 0.5, 0.5])
							])
	img1, img2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
	if use_cuda:
		img1 = img1.cuda()
		img2 = img2.cuda()
	img1, img2 = Variable(img1), Variable(img2)
	out_a, out_b = model(img1), model(img2)
	distance = l2_distance.forward(out_a, out_b)
	print(distance)


if __name__ == '__main__':
	model = ResNet(num_classes=10575, embeddings_num=128, img_size=224, is_fc=True, is_AvgPool=False, pretrained=False)
	model = torch.nn.DataParallel(model).cuda()
	resume = 'weights/resnet34.pth'
	print('[INFO]: loading checkpoint {}'.format(resume))
	checkpoint = torch.load(resume)
	model.load_state_dict(checkpoint['state_dict'])
	test('./pictures/009.jpg', './pictures/011.jpg', model, use_cuda=True)