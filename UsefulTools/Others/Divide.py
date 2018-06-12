# Author: Charles
# Function:
# 	Divide dataset into trainSet and testSet.


f = open('./train.txt', 'r')
train_text_name = './celeba_train.txt'
test_text_name = './celeba_test.txt'
all_pic = f.readlines()
all_count = len(all_pic)
train_count = 182339
f_train = open(train_text_name, 'w')
f_test = open(test_text_name, 'w')
for i in range(train_count):
	f_train.write(all_pic[i])
f_train.close()
for j in range(train_count, all_count):
	f_test.write(all_pic[j])
f_test.close()