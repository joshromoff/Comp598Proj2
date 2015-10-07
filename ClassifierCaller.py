import sys
import pickle
#add classifier directory to runtime paths
sys.path.insert(0,"Classifiers")
#import every classifier
import ExampleClassifier

#we will retrive these from their respective txt file, were created with parseTokenizer.py
trainingSet   = {}
validationSet = {}

#retrive dictionarys with pickle
with open("./DataSetDictionarys/trainingSet.txt","rb") as trainingFile:
    trainingSet = pickle.load(trainingFile)
    
with open("./DataSetDictionarys/validationSet.txt","rb") as validationFile:
    validationSet = pickle.load(validationFile)
  

##################################
#validationResults(fun,validationSet)
#description: run the a particular classification function on the validation set
#input 1: function of type 'tokenized string(list of strings)' -> classification (int)
#input 2: dictionary with key = index (int), value = tokenized string (list of strings)
#output: validation list of type int
def validationResults(fun,validSet):
    classifierVector = []
    #loop through validation set appending the classification as we go
    for i in range(1,len(validSet)):
        classifierVector.append(fun(validSet[i]))
        if i % 100 == 0:
            print 'done ', i

##################################
#trainTestSplit(trainingSet)
#description: splits the training set into training and test (we may want to add k-fold) for now randomize.
#input: training set, dictionary with key = index (int), value = tokenized string (list of strings)
#output: two dictionarys train/test with key = original index, value = tokenized string (list of strings)
def trainTestSplit(trainingTestSplit):
    trainDiv = {}
    testDiv  = {}
    

#Split the training set into training and testing (may want to put this in a loop for k-fold)
(trainDiv,testDiv)             = trainTestSplit(trainingSet)

#Validation results
#function, test error, train error
exFun, exTestErr, exTrainErr   = ExampleClassifier.main(trainDiv,testDiv)
exVector                       = validationResults(exFun,validationSet)

#do some i/o 
    
print classifier1Vector
