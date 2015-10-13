import pickle

with open("./DataSetDictionarys/trainingSetX.txt","rb") as trainingFileX:
    trainingX     = pickle.load(trainingFileX)

print 'done loading training set'

with open("./DataSetDictionarys/trainingSetY.txt","rb") as trainingFileY:
    trainingY       = pickle.load(trainingFileY)

with open("./DataSetDictionarys/validationSet.txt","rb") as validationFile:
    validationSet   = pickle.load(validationFile)

print 'done loading validation set'


def calcDist (l1,l2) :
	return 1
	
	
def findPred (list,k,trainingY):
	index = []
	values = {}
	max = max(list)+1
	for i = 1:k
		max = max(list)
		idx = list.index(min(list))
		list[idx] = max
		index.append(idx)
	mval = 0
	mct = 0
	for id in index:
		v = trainingY(id)
		if v not in values :
			values[v] = 1
			if mct == 0:
				mval = v
				mct = 1
		else :
			values[v] = values[v]+1
			if values[v] > mct:
				mct = values[v]
				mval = v
	return v
		
		
for row in validationSet :
	idx = 1
	distance = [];
	for row2 in trainingX :
		distance.append(calcDist(row,row2))
	pred = findPred(distance,k,trainingY)
	idx = idx+1
	print idx+' '+pred