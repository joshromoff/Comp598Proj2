import nltk
import csv
from random import randint,randrange
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#the goal here is to only have to parse the data once
#save it as dictionarys to be accessed by every classifier we test.

testCSV  = "./DataSetCSVs/ml_dataset_test_in.csv"
trainCSV = "./DataSetCSVs/ml_dataset_train.csv"



#open training csv and parse the data
#format {index,text,class}
f        = open(trainCSV,'rb')
training = csv.reader(f)

trainingCorpus = []
trainingY      = []

#loop through training set, accumulated the text 
for row in training:
    if row[0].isdigit():
        trainingCorpus.append(row[1])
        trainingY.append(int(row[2]))

f.close()

###########################

#cast to vector form
trainingY = np.array(trainingY)

#normalized count vector, could use CountVectorizer() instead
#word needs to appear twice and not be a stop word.
normalizedCountVectorizer = TfidfVectorizer(min_df=2,stop_words='english')

X = normalizedCountVectorizer.fit_transform(trainingCorpus)



fTest            = open(testCSV,'rb')
validation       = csv.reader(fTest)

testingExamples  = [row[1] for row in validation if row[0].isdigit()]

Xtest            = normalizedCountVectorizer.transform(testingExamples)


print "X shape = ", X.shape, "Y shape = ", trainingY.shape , "testX shape = ", Xtest.shape


#pickle it to be opened later
with open("./DataSetWithDictionarys/trainingSetX.txt","wb") as trainFileX:
    pickle.dump(X,trainFileX)

with open("./DataSetWithDictionarys/validationSet.txt","wb") as validationFile:
    pickle.dump(Xtest,validationFile)

with open("./DataSetWithDictionarys/trainingSetY.txt","wb") as trainFileY:
    pickle.dump(trainingY,trainFileY)







    



