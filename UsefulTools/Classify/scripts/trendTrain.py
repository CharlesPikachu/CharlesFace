from pyecharts import Line


def read_txt(filename):
	f = open(filename)
	temp = f.readlines()
	results = []
	for t in temp:
		if t.strip():
			try:
				t = float(t)
				results.append(t)
			except:
				print('[Warning]: Bad data in %s...' % filename)
				continue
	f.close()
	return results


def DrawLine(title, infos):
	line = Line(title)
	attrs = [0]
	values = [0]
	i = 0
	for info in infos:
		attrs.append('Epoch%d' % i)
		values.append(info)
		i += 1
	line.add("Test Acc", attrs, values, is_smooth=False, mark_point=["min", "average", "max"])
	line.render('results.html')



if __name__ == '__main__':
	results = read_txt('./test_acc.txt')
	DrawLine('Step1模型测试性能趋势图', results)