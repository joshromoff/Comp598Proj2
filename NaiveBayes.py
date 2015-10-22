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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

parameters = None
numOutputs = {}
probabOfOutput = {}  

def doLimitedSetSavedData() :

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

def doLimitedSet(useNormal) :
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

    #grab validation set
    print "Reading in Validation csv"
    fTest             = open(testCSV,'rb')
    validation        = csv.reader(fTest)

    testingExamples   = [row[1] for row in validation if row[0].isdigit()]
    fTest.close()

    #VECTORIZE
    print "Extracting features based on the training set"
    vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words='english',max_features=200)
    testingExamples           = vectorizer.fit_transform(testingExamples)
    trainingX                 = vectorizer.transform(trainingCorpus)
    print "NUM FEATURES = ", trainingX.shape[1]
        
    X_train, X_test, Y_train, Y_test = train_test_split(trainingX, trainingY, test_size=0.33, random_state=42)
    
    if useNormal == True :
        features = fitNormalBayes(X_train, Y_train)
        Yprime = predict(X_test, features, useNormal = True)
    else : 
        features = fitBinaryBayes(X_train, Y_train)
        Yprime = predict(X_test, features, useNormal = False)
    print(getSuccessRate(Yprime, Y_test))
    
    nolabel = sklearn.metrics.confusion_matrix(Y_test, Yprime)
    print(nolabel)
    plot_confusion_matrix(nolabel)

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
    
    print('have probabs', probabOfOutput)
    print(data.shape)
    #probabilities of each output is known now
    
    #need to calculate probability of each feature
    #we will model continuous features as normal distributions by output

    splitData = [[],[],[],[]]
    for i in range(np.shape(data)[0]) : #foreach feature/column
        splitData[trainingY[i]].append(data[i])
        #splitData is numCategories x numInputs x numCols
    
    print ('have data')
 #   print np.shape(features)
    print np.shape(splitData)

    features = [[],[],[],[]]
    for i in range(len(splitData)): #for each category
        curData = splitData[i]
        for j in range(curData[0].shape[1]) : #numFeatures
            #get list of the jth feature for all inputs in category
            curCol =  np.array(map(lambda d : d[0,j], curData))
            features[i].append([np.mean(curCol),np.std(curCol)])
    #features is a is 3d list (numCategories, numFeatures, 2)
    print('have features')
    
    return features
    
def fitBinaryBayes(data, trainingY):
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
    
    print('have probabs', probabOfOutput)
    print(data.shape)
    #probabilities of each output is known now
    
    #need to calculate probability of each feature
    #we will model continuous features as normal distributions by output

    splitData = [[],[],[],[]]
    for i in range(np.shape(data)[0]) : #foreach feature/column
        splitData[trainingY[i]].append(data[i])
        #splitData is numCategories x numInputs x numCols
    
    print ('have data')
 #   print np.shape(features)
    print np.shape(splitData)

    features = [[],[],[],[]]
    for i in range(len(splitData)): #for each category
        curData = splitData[i]
        for j in range(curData[0].shape[1]) : #numFeatures
            #get list of the jth feature for all inputs in category
            curCol =  np.array(map(lambda d : d[0,j], curData))
            numNonZero = np.shape(np.nonzero(curCol))[1]
            features[i].append([numNonZero * 1. / np.shape(curCol)[0],np.shape(curCol)[0]])
    #features is a is 3d list (numCategories, numFeatures, 2)
    print('have features')
    
    return features

def predict(data, distrs, useNormal = False):
    #takes in a set of features used for training and return a prediction
    #data is assumed to already be in the form of the features that we are looking at, in this case the words

    data = data.todense()
    print(data.shape)
    
    #distrs is numCats, numFeatures, 2   
    
    Y = []
    for i in range(data.shape[0]) :
        if i % 1000 == 0 :
            print i
        #for each input
        maxProb = None
        toRetOutput = -1
        for output in range(len(distrs)) : # for each category 0 - 3
            probSum = 0
            bias = probabOfOutput[output]
            probSum += np.log(bias)
            for j in range(len(distrs[0])) :
                # for feature
                if useNormal :
                    probSum += normpdf(data[i,j], distrs[output][j][0], distrs[output][j][1])
                else :
                    probSum += binaryPredictAdd(distrs, data[i,j], output, j)
            if maxProb == None or probSum > maxProb :
                maxProb = probSum
                toRetOutput = output
        Y.extend([toRetOutput])

    return Y

def binaryPredictAdd(distrs, dataPoint, category, feature) :
    cats = [0,1,2,3]
    cats.remove(category)
    probGivenCat = distrs[category][feature][0]
    total = 0
    for i in cats :
        num = np.sign(distrs[i][feature][1])
        total += num
    probGivenNotCat = 1
    for i in cats :
        num = distrs[i][feature][1]
        probGivenNotCat += num * 1. / total * np.sign(distrs[i][feature][1])
    
    numerator = probGivenCat
    denominator = probGivenNotCat
    if dataPoint == 0 :
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



def doFullNaiveBayes(smooth_factor) :
    
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


#doFullNaiveBayes(.01)
#doLimitedSetNaiveBayes()
doLimitedSet(True)