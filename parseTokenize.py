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

holdOutSetX    = []
holdOutSetY    = []
#loop through training set, accumulated the text 
for row in training:
    if row[0].isdigit():
        trainingCorpus.append(row[1])
        trainingY.append(int(row[2]))

f.close()

print len(trainingCorpus)
#debugging created holdout set###
count = 0
while count < 1000:
    rando = randint(0,len(trainingCorpus)-1)
    x = trainingCorpus.pop(rando)
    y = trainingY.pop(rando)
    holdOutSetX.append(x)
    holdOutSetY.append(y)
    count +=1

holdOutSetY = np.array(holdOutSetY)
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

##debugging with holdoutset
holdX            = normalizedCountVectorizer.transform(holdOutSetX)
###
print "X shape = ", X.shape, "Y shape = ", trainingY.shape , "testX shape = ", Xtest.shape


#pickle it to be opened later
with open("./DataSetWithHoldSet/trainingSetX.txt","wb") as trainFileX:
    pickle.dump(X,trainFileX)

with open("./DataSetWithHoldSet/validationSet.txt","wb") as validationFile:
    pickle.dump(Xtest,validationFile)

with open("./DataSetWithHoldSet/trainingSetY.txt","wb") as trainFileY:
    pickle.dump(trainingY,trainFileY)


with open("./DataSetWithHoldSet/holdSetX.txt","wb") as holdFileX:
    pickle.dump(holdX,holdFileX)

with open("./DataSetWithHoldSet/holdSetY.txt","wb") as holdFileY:
    pickle.dump(holdOutSetY,holdFileY)
    
#print normalizedCountVectorizer.get_feature_names()




    



