import nltk
import csv
from random import randint
import pickle


#the goal here is to only have to parse the data once
#save it as dictionarys to be accessed by every classifier we test.

testCSV  = "./DataSetCSVs/ml_dataset_test_in.csv"
trainCSV = "./DataSetCSVs/ml_dataset_train.csv"

trainingSet     = {}
validationSet  = {}
        
#open training csv and parse the data
#format {index,text,class}
f        = open(trainCSV,'rb')
training = csv.reader(f)

counter = 0
for row in training:
    if (row[0]).isdigit():
        text       = nltk.word_tokenize(row[1].decode('utf-8'))
        index      = int(row[0])
        classifier = int(row[2])
        if index % 1000 ==0:
            print index ,' of training'

       
        trainingSet[index] = (text,classifier)

    
    
#open test csv and parse/tockenize the data, 
fTest      = open(testCSV,'rb')
validation = csv.reader(fTest)

counter = 0
for row in validation:
    if (row[0]).isdigit():
        text       = nltk.word_tokenize(row[1].decode('utf-8'))
        index      = int(row[0])
        if index % 1000 ==0:
            print index ,' of validation'
        validationSet[index] = text

   

#dump the dictionarys using pickle so we dont have to reparse every time
with open("./DataSetDictionarys/trainingSet.txt","wb") as trainFile:
    pickle.dump(trainingSet,trainFile)

with open("./DataSetDictionarys/validationSet.txt","wb") as validationFile:
    pickle.dump(validationSet,validationFile)
    
print len(trainingSet), ',' ,len(validationSet)
    

