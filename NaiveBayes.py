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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

parameters = None
numOutputs = {}
probabOfOutput = {}  

def doLimitedSetNaiveBayes() :

    X=None
    Xtest=None
    trainingY=None
        
    #we use the count vectorizer output, not the tf-idf output
    with open("./DataSetDictionarys/trainingSetX.txt","rb") as trainFileX:
        X = pickle.load(trainFileX)
    
    with open("./DataSetDictionarys/validationSet.txt","rb") as validationFile:
        Xtest = pickle.load(validationFile)
    
    with open("./DataSetDictionarys/trainingSetY.txt","rb") as trainFileY:
        trainingY = pickle.load(trainFileY)

    #K best features
    KBESTNUM = 100
    
    print "ORIGINAL #FEATURES", X.shape[1]
    print "removing features with zero variance"
    sel = VarianceThreshold(threshold = (.000005))
    X = sel.fit_transform(X)
    print "feature selecting using " +str(KBESTNUM)
    
    kBest         = SelectKBest(f_classif,k=KBESTNUM)
    X     = kBest.fit_transform(X,trainingY)
    
    print "done selecting features"
    
    skf = sklearn.cross_validation.StratifiedKFold(trainingY, n_folds=5)
    
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = trainingY[train_index], trainingY[test_index]

        features = fitNormalBayes(X_train, y_train)
        Yprime = predict(X_test, features, useNormal = True)
        print(getSuccessRate(Yprime, y_test))

def doLimitedWithReloading() :
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
    
    features = fitNormalBayes(trainingX, trainingY)
    Yprime = predict(trainingX, features, useNormal = True)
    print(getSuccessRate(Yprime, trainingY))

def fitNormalBayes(data, trainingY):
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
    for i in range(np.shape(data)[1]) : #foreach feature/column
        featureInstance = [[], [], [], []]
        col = data.getcol(i)
        nonZeros = col.nonzero()
        for index in range(len(nonZeros[0])) :
            #all the cols should be 0, one column
            featureInstance[trainingY[nonZeros[0][index]]].extend(col[index][0].todense())
        features.append(featureInstance)
    # features is [numCols, numCategories, numInputsInCategory]
    print ('have features')
    print np.shape(features)
    for feature in features :
        byFeature = []
        #print np.shape(feature)
        for i in range(np.shape(feature)[0]) :
            outputVals = featureInstance[i]
            #distribution of data in this category
            numNonZeros = len(outputVals)
            numOutputsOfCateg = numOutputs[i]
            arrWithZeros = np.append(np.array(outputVals), np.zeros((numOutputsOfCateg - numNonZeros, 1)))
            mean = np.mean(arrWithZeros)
            std = np.std(arrWithZeros)
            byFeature.append([mean, std])
        distrs.append(byFeature)
    #distrs is 3d list (numFeatures, numOutputs, 4)
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

def predict(data, distrs, useNormal = False):
    #takes in a set of features used for training and return a prediction
    #data is assumed to already be in the form of the features that we are looking at, in this case the words

    data = data.todense()
    print(data.shape)
    
    Y = []
    for i in range(data.shape[0]) :
        #for each input
        #print i, data.shape[0], np.shape(distrs)
        maxProb = None
        toRetOutput = -1
        for output in range(len(distrs[1])) : # for each category 0 - 3
            probSum = 0
            #p = probabOfOutput[output]
            #bias = np.log(p) - np.log(1-p)
            #probSum += bias
            for j in distrs :
                # for each feature
                if useNormal :
                    probSum += normalPredictAdd(output, data, i, j)
                else :
                    probSum += binaryPredictAdd(output, data, i, j)
            if maxProb == None or probSum > maxProb :
                maxProb = probSum
                toRetOutput = output
        Y.extend([toRetOutput])

    return Y

def normalPredictAdd(output, data, i, j) :
    prob = normpdf(data[i,output], j[output][0], j[output][1])
    #probGivenNot = normpdf(data[i,output], j[output][2], j[output][3])         
    return prob #- np.log(probGivenNot)

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
def normpdf(x, mean, std):
    var =  math.pow(std, 2)
    denom = math.pow((2*math.pi*var), .5)
    num = math.exp(- math.pow((x-mean),2)/(2*var))
    return np.log(num) - np.log(denom)

def gaussianpdf(x, mean, std):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exponent

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





def doFullNaiveBayes() :
    
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

    smooth_factor = .01

    probabilities, categoryRatio, vocabSize, corpusSizeInCat = trainLaplace(X_train, y_train, smooth_factor)
    YPrime = makeFullPredictions(probabilities, vocabSize, corpusSizeInCat, categoryRatio, smooth_factor, X_test, y_test)
    getSuccessRate(YPrime, y_test)
            
def trainLaplace(X_train, y_train, smooth_factor) :
    
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
    
    allString = "0123"
    corpusSizeWithoutCategory = []
    for i in range(4) :
        corpusSizeWithoutCategory.extend([sum(numIn) - numIn[i] + vocabSize * smooth_factor])
        
    for i in range(4) :
        #we find the probabilities for each word
        for word in range(X.shape[1]) : #for each word/feature
            if word % 1000 == 0 :
                print word, i
            #use laplace smoothing and estimate values
            #print X.shape, np.shape(numIn), type(numIn[i]), numIn[i]+vocabSize, word, i
            probabilities[(featureNames[word], str(i))] = (X[i, word] + smooth_factor) / (numIn[i] + vocabSize * smooth_factor)
            probabilities[(featureNames[word], allString.replace(str(i), ''))] = (np.array(sp.sparse.csr_matrix.sum(X, 0))[0][word] - X[i, word] + smooth_factor) / corpusSizeWithoutCategory[i]
    
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

def makeFullPredictions(probabilities, vocabSize, corpusSizeInCat, categoryRatio, smooth_factor, X_test, y_test) :
    allString = '0123'
    Y = []
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
                    prob = prob = (np.log(smooth_factor) - np.log(corpusSizeInCat[i] + vocabSize * smooth_factor)) - (np.log(smooth_factor) - np.log(sum(corpusSizeInCat) - corpusSizeInCat[i] + vocabSize * smooth_factor))
                if math.isinf(prob) or math.isnan(prob) :
                    prob = 0
                probSum += prob
            totalProb = bias + probSum
            if totalProb > maxProb :
                maxProb = totalProb
                toRetOutput = i
        Y.extend([toRetOutput])
        
    return Y


#doFullNaiveBayes()
#doLimitedSetNaiveBayes()
doLimitedWithReloading()