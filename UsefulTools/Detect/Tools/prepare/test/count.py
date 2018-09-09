import os


total = 0
for file in os.listdir('./images'):
	total += len(os.listdir('./images' + '/' + file))
print(len(os.listdir('./images')))
print(total)