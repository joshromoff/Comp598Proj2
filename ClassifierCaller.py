import pickle
from scipy import sparse
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble         import BaggingClassifier
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.naive_bayes      import MultinomialNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import Perceptron



#retrieve data with pickle
with open("./DataSetDictionarys/trainingSetX.txt","rb") as trainingFileX:
    trainingX     = pickle.load(trainingFileX)

print 'done loading training set'

with open("./DataSetDictionarys/trainingSetY.txt","rb") as trainingFileY:
    trainingY       = pickle.load(trainingFileY)

with open("./DataSetDictionarys/validationSet.txt","rb") as validationFile:
    validationSet   = pickle.load(validationFile)

print 'done loading validation set'


#Classifier list, to be iterated over
classifiers = [
    #KNeighborsClassifier(3),
    MultinomialNB(),
    Perceptron(),
    BaggingClassifier(Perceptron(),n_estimators = 100,max_samples=0.5, max_features=0.5)
    ]

names = ["MultinomialNB","Perceptron","BaggingPerceptron"]




#training test splitting
X_train, X_test, y_train, y_test = train_test_split(trainingX, trainingY, test_size=.4)

print "train test split done"

for name,clf in zip(names,classifiers):
    print "Training using " + name + " classifier"
    clf.fit(X_train,y_train)
    print name," fitted"
    score = clf.score(X_test,y_test)
    print name , score

#############################

#predict actual results!
for name,clf in zip(names,classifiers):
    print "Prediction using " + name + " classifier"
    clf.fit(trainingX,trainingY)
    print name," fitted"
    validationPredictions = clf.predict(validationSet)
    with open(name + 'predictions.csv','wb') as predictions:
        writer = csv.writer(predictions)
        writer.writerow(['Id','Prediction'])
        counter = 0
        for prediction in validationPredictions:
            writer.writerow([counter,prediction])
            counter += 1
    
