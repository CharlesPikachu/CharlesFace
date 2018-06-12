# Author: Charles
# Function:
# 	Generate train pic path.
import os


save_txt_name = 'train.txt'
Image_path = os.getcwd() + '/images'
imgs_list = sorted(os.listdir(Image_path))
fp = open(save_txt_name, 'w')
for img in imgs_list:
	fp.write(Image_path + '/' + img + '\n')
fp.close()