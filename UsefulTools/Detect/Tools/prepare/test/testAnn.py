import os

path = r'C:\Users\ThinkPad\Desktop\CelebA_face_detect\labels'
for p in sorted(os.listdir(path)):
	with open(path+'\\'+p, 'r') as f:
		t = f.readlines()
		print(len(t))
		if len(t) > 1:
			print(p)
			break
	f.close()