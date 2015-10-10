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
from sklearn.linear_model     import LogisticRegression
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.svm              import LinearSVC
from sklearn.ensemble         import RandomForestClassifier


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
    #DecisionTreeClassifier(max_features=.2,max_depth =3),
    LinearSVC(),
    BaggingClassifier(LinearSVC(),n_estimators = 50,max_samples=0.5, max_features=0.5),
    BaggingClassifier(Perceptron(),n_estimators = 50,max_samples=0.5, max_features=0.5)
    ]

names = ["MultinomialNB","Perceptron","SVC","BaggingSVC","BaggingPerceptron"]

#SET THIS FLAG TO TRUE IF YOU WANT TO PREDICT THE VALIDATION SET, NO SCORE FOR METAS 
DOING_PREDICTIONS = True

#if we arent doing predictions then we use a prediction set to estimate our meta classifiers
#training/test validation splitting, hold out 10% for metta score.
if not DOING_PREDICTIONS:
    trainingX, X_validate, trainingY, y_validate = train_test_split(trainingX, trainingY, test_size=.1)

#split training and splitting 60/40 of 90%
trainingX, X_test, trainingY, y_test = train_test_split(trainingX, trainingY, test_size=.3)
print "train test split done"

metaTrainingSet = []
metaPredictionSet = []
for name,clf in zip(names,classifiers):
    print  name + " Training"
    clf.fit(trainingX,trainingY)
    print "fitted"
    #if we arent doing predictions then we use a prediction set to estimate our meta classifiers
    if not DOING_PREDICTIONS:
        metaPredictionSet.append(clf.predict(X_validate))
        score = clf.score(X_test,y_test)
        print name , score
    metaTrainingSet.append(clf.predict(X_test))
    


##predict actual results!
if DOING_PREDICTIONS:
    for name,clf in zip(names,classifiers):
        print "Prediction using " + name + " classifier"
        validationPredictions = clf.predict(validationSet)
        metaPredictionSet.append(validationPredictions)
        with open('./Predictions/'+ name+ 'predictions.csv','wb') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 0
            for prediction in validationPredictions:
                writer.writerow([counter,prediction])
                counter += 1

#############################
#meta classifier that combines them all

metaTrainingSet = np.transpose(np.array(metaTrainingSet))
metaPredictionSet = np.transpose(np.array(metaPredictionSet))

metaClassifiers = [ BaggingClassifier(n_estimators = 500,max_samples=0.5, max_features=0.5),
                    AdaBoostClassifier(n_estimators = 500,algorithm='SAMME'),
                    RandomForestClassifier(n_estimators=1000)
                    ]
metaNames = ["Bagging","AdaBoost","RandomForest"]

print "META TRAINING"
for name,meta in zip(metaNames,metaClassifiers):                
    print "META " + name
    meta.fit(metaTrainingSet,y_test)
    print "fitted"
    if not DOING_PREDICTIONS:
        score = meta.score(metaPredictionSet,y_validate)
        print "Meta" + name , score





#do metas
if DOING_PREDICTIONS:
    for name,clf in zip(metaNames,metaClassifiers):
        print "Prediction using " + name + " classifier"
        validationPredictions = clf.predict(metaPredictionSet)
        with open('./Predictions/'+ name+ 'predictions.csv','wb') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 0
            for prediction in validationPredictions:
                writer.writerow([counter,prediction])
                counter += 1






    
