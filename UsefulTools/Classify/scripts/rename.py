# rename script.
import os


i = 0
for file in sorted(os.listdir('.')):
	if file[-2:] == 'py':
		continue
	if i < 10:
		os.rename(file, '0000' + str(i))
		i += 1
		continue
	elif i < 100:
		os.rename(file, '000' + str(i))
		i += 1
		continue
	elif i < 1000:
		os.rename(file, '00' + str(i))
		i += 1
		continue
	elif i < 10000:
		os.rename(file, '0' + str(i))
		i += 1
		continue
	elif i < 100000:
		os.rename(file, str(i))
		i += 1
		continue