# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:45:18 2015

@author: Eric
"""

import numpy as np
import pickle
import math
import sklearn
import scipy as sp
import csv
import string

parameters = None
numOutputs = {}
probabOfOutput = {}  

def fitTFIDFNaiveBayes(data, trainingY):
    #data is expected to be a matrix with the final columns as the outputs
    #needs to figure out the probabilities given the output  
    
    #find out probability of each output
    total = 0
    for i in range(data.shape[0]) :
        output = trainingY[i]
        if output not in numOutputs :
            numOutputs[output] = 0
        numOutputs[output] += 1 
        total += 1
    
    for key in numOutputs.keys() :
        probabOfOutput[key] = numOutputs[key] * 1.0 / total
    
    print('have probabs')
    print(data.shape)
    #probabilities of each output is known now
    
    #need to calculate probability of each feature
    #we will model continuous features as normal distributions by output
    features = [] #list of inputs by feature by output   
    distrs = []
    for i in range(data.shape[1]) : #foreach feature/column
        #print i
        byFeature = []
        featureInstance = [[], [], [], []]
    
        #tf-idf
        col = data.getcol(i)
        nonZeros = col.nonzero()
        for index in range(len(nonZeros[0])) :
            #all the cols should be 0, one column
            featureInstance[trainingY[nonZeros[0][index]]].extend(col[index][0].todense())


        features.append(featureInstance)
    # features is [numCols, numCategories, numInputsInCategory]
    
    print ('have features')
    
    #tf-idf only
    print np.shape(features)
    for feature in features :
        byFeature = []
        #print np.shape(feature)
        for i in range(np.shape(feature)[0]) :
            outputVals = featureInstance[i]
            #distribution of data in this category
            numNonZeros = len(outputVals)
            numOutputsOfCateg = numOutputs[i]
            numOtherCats = data.shape[0] - numOutputsOfCateg

            #print (type(outputVals))
            #print np.shape(outputVals)
            mean = np.mean(np.append(np.array(outputVals), np.zeros((numOutputsOfCateg - numNonZeros, 1))))
            std = np.std(np.append(np.array(outputVals), np.zeros((numOutputsOfCateg - numNonZeros , 1))))
            
            #distribution of data in other categories
            others = []
            for category in feature :
                others.extend([datum for datum in category if (category != outputVals)])
            meanNotOutput = np.mean(np.append(np.array(others), np.zeros((numOtherCats - len(others) , 1))))
            stdNotOutput = np.std(np.append(np.array(others), np.zeros((numOtherCats - len(others) , 1))))
            
            byFeature.append([mean, std, meanNotOutput, stdNotOutput])
  
        distrs.append(byFeature)

    #tf-idf distrs is 3d list (numFeatures, numOutputs, 4)
    print('have distrs')
    
    return distrs

def fitBinaryNaiveBayes(data, trainingY):
    #data is expected to be a matrix with the final columns as the outputs
    #needs to figure out the probabilities given the output  
    
    #find out probability of each output
    total = 0
    for i in range(data.shape[0]) :
        output = trainingY[i]
        if output not in numOutputs :
            numOutputs[output] = 0
        numOutputs[output] += 1 
        total += 1
    
    for key in numOutputs.keys() :
        probabOfOutput[key] = numOutputs[key] * 1.0 / total
    
    print('have probabs')
    print(data.shape)
    #probabilities of each output is known now
    
    #need to calculate probability of each feature
    #we will model continuous features as normal distributions by output
    features = [] #list of inputs by feature by output
    data = data.todense()    
    distrs = []
    for i in range(data.shape[1]) : #foreach feature/column
        #print i
        byFeature = []
        featureInstance = [[], [], [], []]
        
        #binary
        for j in range(4) :
            dataI = data[trainingY[:] == j , i] #get all data with the correct category
            dataNotI = data[trainingY[:] != j , i]
            featureInstance[j] = dataI
            
            #get p of token existing or not
            onesCat = dataI[dataI[:] > 0]
            onesNotCat = dataNotI[dataNotI[:] > 0]
            probGivenCat = onesCat.shape[1] * 1. /  dataI.shape[0]
            probGivenNotCat =  onesNotCat.shape[1] * 1. /  dataI.shape[0]
            byFeature.append([probGivenCat, probGivenNotCat])
        distrs.append(byFeature)

        features.append(featureInstance)
    # features is [numCols, numCategories, numInputsInCategory]
    
    print ('have features')
    
    #binary distrs is 3d lsit [numFeatures, numCategories, 2]
    print('have distrs')
    
    return distrs

def predict(data, distrs, usetfidf = False):
    #takes in a set of features used for training and return a prediction
    #data is assumed to already be in the form of the features that we are looking at, in this case the words

    data = data.todense()
    print(data.shape)
    
    Y = []
    for i in range(data.shape[0]) :
        #for each input
        #print i, data.shape[0], np.shape(distrs)
        maxProb = -1e300
        toRetOutput = -1
        for output in range(len(distrs[1])) :
            p = probabOfOutput[output]
            bias = np.log(p/(1-p))
            probSum = 0
            for j in distrs :
                # for each feature
                if usetfidf :
                    probSum+=tfidfPredictAdd(output, data, i, j)
                else :
                    probSum += binaryPredictAdd(output, data, i, j)
            totalProb = bias + probSum
            if totalProb > maxProb :
                maxProb = totalProb
                toRetOutput = output
        Y.extend([toRetOutput])
        
    return Y

def tfidfPredictAdd(output, data, i, j) :
    prob = normpdf(data[i,output], j[output][0], j[output][1])
    probGivenNot = normpdf(data[i,output], j[output][2], j[output][3])         
    return np.log(prob) - np.log(probGivenNot)

def binaryPredictAdd(output, data, i, j) :
    probGivenCat = j[output][0]
    probGivenNotCat = j[output][1]

    numerator = probGivenCat
    denominator = probGivenNotCat
    if data[i, output] == 0 :
        numerator = 1 - numerator
        denominator = 1 - denominator
    if denominator == 0 :
        return 0
    return np.log(numerator)  - np.log(denominator)

#source : http://stackoverflow.com/questions/12412895/calculate-probability-in-normal-distribution-given-mean-std-in-python
def normpdf(x, mean, sd):
   # print "x, mean, sd", x, mean, sd
    var = sd**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(x-mean)**2/(2*var))
    return num/denom    

def getRowIndex(data, row) :
    #didn't end up using, leaving in in case I need it
    #data as a sparse matrix, row as a row of a sparse matrix   
    for i in range(data.shape[0]):
        rowZ2 = data[i].eliminate_zeros()
        rowZ = row.eliminate_zeros()
        if np.array_equal(rowZ.todense(), rowZ2.todense()) :
            return
        return False

def getSuccessRate(guess, true) :
    correct = 0
    total = 0
    for i in range(len(guess)) :
        if guess[i] == true[i] :
            correct += 1
        total +=1

    return correct * 1.0 / total    

def getCorpusData() :
    
    testCSV  = "./DataSetCSVs/ml_dataset_test_in.csv"
    trainCSV = "./DataSetCSVs/ml_dataset_train.csv"
    
    f = open(trainCSV,'rb')
    training = csv.reader(f)

    trainingCorpus = []
    trainingY      = []

    #loop through training set, accumulated the text 
    for row in training:
        if row[0].isdigit():
            trainingCorpus.append(row[1])
            trainingY.append(int(row[2]))
            
    skf = sklearn.cross_validation.StratifiedKFold(trainingY, n_folds=4)

    trainingCorpus = np.array(trainingCorpus)
    trainingY = np.array(trainingY)

    for train_index, test_index in skf :
        X_train, X_test = trainingCorpus[train_index], trainingCorpus[test_index]
        y_train, y_test = trainingY[train_index], trainingY[test_index]

    probabilities, categoryRatio, vocabSize, corpusSizeInCat = trainLaplace(X_train, y_train)
    
    allString = '0123'
    correct = 0
    total = 0
    for row in range(len(X_test)):
        line = "".join([ch for ch in X_test[row] if ch not in string.punctuation])
        toRetOutput = -1
        maxProb = -1e300
        for i in range(4) :
            p = categoryRatio[i]
            bias = np.log(p/(1-p))
            probSum = 0
            for word in line.split(" ") :
                # for each feature
                if (word, i) in probabilities :
                    prob = np.log(probabilities(word, i)) -  np.log(probabilities(word, allString.replace(str(i), '')))                        
                else :
                    prob = prob = (np.log(1 ) - np.log(corpusSizeInCat[i] + vocabSize)) - (np.log(1) - np.log(sum(corpusSizeInCat) - corpusSizeInCat[i] + vocabSize))
                probSum += prob
            totalProb = bias + probSum
            if totalProb > maxProb :
                maxProb = totalProb
                toRetOutput = i
        guess = toRetOutput
        if guess == y_test[row] :
            correct += 1
        total += 1
    print correct * 1. / total    
            
def trainLaplace(X_train, y_train) :
    
    trainCorpus = ["","","",""]
    for i in range(len(X_train)) :
        trainCorpus[y_train[i]] += " " + X_train[i]

    for i in range(len(trainCorpus)) :
        trainCorpus[i] = "".join([ch for ch in trainCorpus[i] if ch not in string.punctuation])

    countVectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X = countVectorizer.fit_transform(trainCorpus)

    numIn  = []
    for num in range(4) :
        numIn.extend(np.array(sp.sparse.csr_matrix.sum(X, 1)[num])[0])

    vocabSize = X.shape[1]
    
    probabilities = {}
    featureNames = countVectorizer.get_feature_names()    
    print featureNames
    
    allString = "0123"
    corpusSizeWithoutCategory = []
    for i in range(4) :
        corpusSizeWithoutCategory.extend([sum(numIn) - numIn[i] + vocabSize])
        
    for i in range(4) :
        #we find the probabilities for each word
        for word in range(X.shape[1]) : #for each word/feature
            if word % 1000 == 0 :
                print word, i
            #use laplace smoothing and estimate values
            #print X.shape, np.shape(numIn), type(numIn[i]), numIn[i]+vocabSize, word, i
            probabilities[(featureNames[word], str(i))] = (X[i, word] + 1.) / (numIn[i] + vocabSize)
            probabilities[(featureNames[word], allString.replace(str(i), ''))] = (np.array(sp.sparse.csr_matrix.sum(X, 0))[0][word] - X[i, word] + 1.) / corpusSizeWithoutCategory[i]
    
    categoryRatio = {}
    totalOutputs = {}
    total = 0
    for i in range(trainingY.shape[0]) :
        output = trainingY[i]
        if output not in totalOutputs :
            totalOutputs[output] = 0
        totalOutputs[output] += 1 
        total += 1
    
    for key in totalOutputs.keys() :
        categoryRatio[key] = totalOutputs[key] * 1.0 / total    
    
    return probabilities, categoryRatio, vocabSize, numIn

X=None
Xtest=None
trainingY=None
    
#we use the count vectorizer output, not the tf-idf output
with open("./DataSetDictionarys/trainingSetX2.txt","rb") as trainFileX:
    X = pickle.load(trainFileX)

with open("./DataSetDictionarys/validationSet2.txt","rb") as validationFile:
    Xtest = pickle.load(validationFile)

with open("./DataSetDictionarys/trainingSetY2.txt","rb") as trainFileY:
    trainingY = pickle.load(trainFileY)


getCorpusData()

#from sklearn.feature_selection import VarianceThreshold
#from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import SelectKBest

##K best features
#KBESTNUM = 200
#
#print "ORIGINAL #FEATURES", X.shape[1]
#print "removing features with zero variance"
#sel = VarianceThreshold(threshold = (.000005))
#X = sel.fit_transform(X)
#print "feature selecting using " +str(KBESTNUM)
#
#kBest         = SelectKBest(f_classif,k=KBESTNUM)
#X     = kBest.fit_transform(X,trainingY)
#
#print "done selecting features"
#
#skf = sklearn.cross_validation.StratifiedKFold(trainingY, n_folds=4)
#
#for train_index, test_index in skf:
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = trainingY[train_index], trainingY[test_index]
#
#    features = fitBinaryNaiveBayes(X_train, y_train)
#    Yprime = predict(X_test, features)
#    print(getSuccessRate(Yprime, y_test))