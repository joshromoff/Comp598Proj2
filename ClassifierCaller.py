import sys
import pickle

#add classifier directory to runtime paths
sys.path.insert(0,"Classifiers")

#import every classifier
from AbstractClassifier import AbstractClassifier
from TestClassifier     import TestClassifier






##################################
#validationResults(fun,validationSet)
#description: run the a particular classification function on the validation set
#input 1: Classifier Object that implements AbstractClassifier
#input 2: dictionary with key = index (int), value = (tokenized string (list of strings), class)
#output: validation list of type int
def classifySet(classifier,Set):
    classifierDict = {}
    #loop through validation set appending the classification as we go
    for key,value in Set.iteritems():
        text,classif = value
        classifierDict[key] = classifier.Classify(text)
    return classifierDict



##################################
#trainTestSplit(trainingSet)
#description: splits the training set into training and test (we may want to add k-fold)
#input: training set, dictionary with key = index (int), value = (tokenized string (list of strings),class)
#output: two dictionarys train/test with key = original index, value = (tokenized string (list of strings),class)
def trainTestSplit(trainingSet):
    trainDiv = {key: value for i, (key, value) in enumerate(trainingSet.viewitems()) if i % 2 == 0}
    testDiv  = {key: value for i, (key, value) in enumerate(trainingSet.viewitems()) if i % 2 == 1}

    return(trainDiv,testDiv)


#calculates the difference in the actual data set vs a classifiers prediction
#they should have identical keys! (we could use lists or a numpy array to make this faster, (don't think its necessary)
def totalError(dataSet,predictions):
    if len(set(dataSet.keys()) - set(predictions.keys())) != 0: #this shouldnt happen!
        print 'ERROR'
        return 0
    
    totalError = 0
    for key in dataSet:
        text1,actual = dataSet[key]
        predicted    = predictions[key]
        if actual != predicted:
            totalError += 1
            
    return totalError




#MAIN#
#1)reads in the already parsed data
#2)splits the training set (still needs work for k fold)
#3)Evaluates a particular Classifier


#we will retrieve these from their respective txt file, were created with parseTokenizer.py
trainingSet         = {}
validationSet       = {}

#retrieve dictionarys with pickle
with open("./DataSetDictionarys/trainingSet.txt","rb") as trainingFile:
    trainingSet     = pickle.load(trainingFile)

print 'done loading training set'

with open("./DataSetDictionarys/validationSet.txt","rb") as validationFile:
    validationSet   = pickle.load(validationFile)

print 'done loading validation set'

#Split the training set into training and testing (may want to put this in a loop for k-fold)
(trainDiv,testDiv)  = trainTestSplit(trainingSet)


#EXAMPLE of how to evaluate a Classifier for a given training/test split#
#Object containing a function to classify an example.
exampleClassifier   = TestClassifier(trainDiv,testDiv)

#Calculate our training error, need to create a prediction vector for our training set
exampleTrainPredict = classifySet(exampleClassifier,trainDiv)
trainingError       = totalError(trainDiv,exampleTrainPredict)

#Calculate our test error, need to create a prediction vector for our test set
exampleTestPredict  = classifySet(exampleClassifier,testDiv)
testError           = totalError(testDiv,exampleTestPredict)

#with the classifier create the vector with our predictions
tempValidationSet   = {key: (text,-1) for key,text in validationSet.iteritems()} #converts validation dictionary to a tuple dictionary to use with classifySet. this could be removed if we reparse the data 
exampleValid        = classifySet(exampleClassifier,tempValidationSet)

print testError, ',' ,trainingError

    

