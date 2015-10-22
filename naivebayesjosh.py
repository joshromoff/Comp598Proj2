# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics                 import accuracy_score
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


def separateByClass(dataset,datasetY):
	separated = {}
	for i in range(dataset.shape[0]):
		vector = dataset[i]
		classValue = datasetY[i]
		if (classValue not in separated):
			separated[classValue] = []
		separated[classValue].append(i)
	return separated

def summarize(dataIndices,dataset):
        print(type(dataset))
        curData = dataset[dataIndices]
        summaries = []
        for i in range(dataset.shape[1]):
            curCol =  curData[:,i].toarray()
            summaries.append((curCol.mean(),curCol.std()))
	return summaries

def summarizeByClass(dataset,datasetY):
	separated = separateByClass(dataset,datasetY)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances,dataset)
	print "done all summaries"
	return summaries

def calculateProbability(x, averg, std):
	exponent = math.exp(-(math.pow(x-averg,2)/(2*math.pow(std,2))))
	return (1 / (math.sqrt(2*math.pi) * std)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 0
		for i in range(len(classSummaries)):
			averg, std = classSummaries[i]
			
			x = inputVector[0,i]
            
			probabilities[classValue] += np.log(calculateProbability(x, averg, std))
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(testSet.shape[0]):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	return accuracy_score(testSet,predictions)

def main():
    ################################################
    #csv data extraction
    testCSV           = "./DataSetCSVs/ml_dataset_test_in.csv"
    trainCSV          = "./DataSetCSVs/ml_dataset_train.csv"



    #open training csv and parse the data
    #format {index,text,class}
    f                 = open(trainCSV,'rb')
    training          = csv.reader(f)

    trainingCorpus    = []
    trainingY         = []

    #loop through training set, accumulated the text
    print "Reading in Training csv"
    for row in training:
        if row[0].isdigit():
            trainingCorpus.append(row[1])
            trainingY.append(int(row[2]))

    f.close()
    #cast to vector form
    trainingY         = np.array(trainingY)
    trainingCorpus    = np.array(trainingCorpus)

    #trainingCorpus = trainingCorpus.reshape(len(trainingCorpus),1)


    #grab validation set
    print "Reading in Validation csv"
    fTest             = open(testCSV,'rb')
    validation        = csv.reader(fTest)

    testingExamples   = [row[1] for row in validation if row[0].isdigit()]
    fTest.close()

    #VECTORIZE
    print "Extracting features based on the training set"
    normalizedCountVectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words='english',max_features=100)
    testingExamples           = normalizedCountVectorizer.fit_transform(testingExamples)
    trainingX                 = normalizedCountVectorizer.transform(trainingCorpus)
    print "NUM FEATURES = ", trainingX.shape[1]

    print type(trainingX)
    # prepare model
    summaries = summarizeByClass(trainingX,trainingY)
    # test model
    predictions = getPredictions(summaries, trainingX)
    accuracy = getAccuracy(trainingY, predictions)
    print('Accuracy: {0}%').format(accuracy)

main()