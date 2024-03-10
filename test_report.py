import numpy
datasets = ['Pubmed', 'Cora', 'Citeseer']
res_test = dict()
res_dev = dict()
settings = []
for dataset in datasets:
	with open(f'what_matters_{dataset}', 'r') as fp:
		lines = fp.readlines()
	for i, line in enumerate(lines):
		if line.startswith(dataset):
			dev, test = lines[i+1].split(', ')
			try:
				res_dev[dataset].append(float(dev))
				
				res_test[dataset].append(float(test))
				settings.append(line)
			except:
				res_dev[dataset] = []
				res_test[dataset] = []

	i = numpy.array(res_dev[dataset]).argmax()
	print(numpy.array(res_test[dataset])[i], settings[i], dataset)
