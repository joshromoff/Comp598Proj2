# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:45:18 2015

@author: Eric
"""

import numpy as np
import pickle
import math

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
        print i
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
        print i
        byFeature = []
        featureInstance = [[], [], [], []]
        
        #binary
        for j in range(4) :
            dataI = data[trainingY[:] == j , i] #gets all non 0 values
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
    
    data = data.todense()    
    print(data.shape)
    
    Y = []
    for i in range(data.shape[0]) :
        #for each input
        print i, data.shape[0], np.shape(distrs)
        maxProb = 1e-300
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
                toRetOutput = output
        Y.extend([toRetOutput])
        
    return Y

def tfidfPredictAdd(output, data, i, j) :
    prob = normpdf(data[i,output], j[output][0], j[output][1])
    probGivenNot = normpdf(data[i,output], j[output][2], j[output][3])         
    return np.log(prob/probGivenNot)

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
    return np.log(numerator * 1.0 /denominator)

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

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

#K best features
KBESTNUM = 400
 
print "ORIGINAL #FEATURES", X.shape[1]
print "removing features with zero variance"
sel = VarianceThreshold(threshold = (.000005))
X = sel.fit_transform(X)
print "feature selecting using " +str(KBESTNUM)

kBest         = SelectKBest(f_classif,k=KBESTNUM)
X     = kBest.fit_transform(X,trainingY)
 
print "done selecting features"

features = fitBinaryNaiveBayes(X, trainingY)
Yprime = predict(X, features)
print(getSuccessRate(Yprime, trainingY))
