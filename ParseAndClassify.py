import pickle
from scipy import sparse
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble         import BaggingClassifier
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.naive_bayes      import MultinomialNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors        import LSHForest
from sklearn.linear_model     import Perceptron
from sklearn.linear_model     import LogisticRegression
from sklearn.linear_model     import LinearRegression
from sklearn.linear_model     import RandomizedLogisticRegression
from sklearn.linear_model     import SGDClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.linear_model     import PassiveAggressiveClassifier
from sklearn.svm              import LinearSVC
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import RandomizedLogisticRegression
from sklearn.neighbors import KDTree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
#from sklearn.cross_validation import cross_val_predict_proba
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
#FLAGS#
DOING_PREDICTIONS = True
RUN_METAS         = True
RUN_REGS          = True
RUN_WD            = True
#K best features, (1,3) gram = 30k,(1,2) = 25k,(1,1)=7.5k
KBESTNUM = 50000
KBESTNUMWD = 10
#K FOLD
KFOLD  = 5

#GRAM "1gram =(1,1)","2gram=(2,2)","12gram=(1,2)","3gram=(3,3)","123gram=(1,3)"
GRAM = (1,1)
################################################
#csv data extraction
testCSV  = "./DataSetCSVs/ml_dataset_test_in.csv"
trainCSV = "./DataSetCSVs/ml_dataset_train.csv"



#open training csv and parse the data
#format {index,text,class}
f        = open(trainCSV,'rb')
training = csv.reader(f)

trainingCorpus = []
trainingY      = []

#loop through training set, accumulated the text
print "Reading in Training csv"
for row in training:
    if row[0].isdigit():
        trainingCorpus.append(row[1])
        trainingY.append(int(row[2]))

f.close()
#cast to vector form
trainingY = np.array(trainingY)
trainingCorpus = np.array(trainingCorpus)

#trainingCorpus = trainingCorpus.reshape(len(trainingCorpus),1)


#grab validation set
print "Reading in Validation csv"
fTest            = open(testCSV,'rb')
validation       = csv.reader(fTest)

testingExamples  = [row[1] for row in validation if row[0].isdigit()]
fTest.close()


############################################
#Classifier list, to be iterated over
classifiers = [
    #MultinomialNB(),
    #Perceptron(),
    #KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',leaf_size=1),
    #DecisionTreeClassifier(max_depth=500,max_features=.05),
    SGDClassifier(),
    PassiveAggressiveClassifier(),
    LinearSVC(),
    #LSHForest()
    #BaggingClassifier(SVC(cache_size = 500,degree = 2),n_estimators = 100,max_samples=0.001, max_features=0.5)
    LogisticRegression(max_iter = 100, solver = 'newton-cg'),
    #BaggingClassifier(LinearSVC(),n_estimators = 100,max_samples=0.5, max_features=0.5),
    BaggingClassifier(Perceptron(),n_estimators = 100,max_samples=0.5, max_features=0.5),
    #BaggingClassifier(SGDClassifier(),n_estimators = 100,max_samples=0.5, max_features=0.5)
    #AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 1000,algorithm='SAMME.R')
    ]


names = [#("MultinomialNB",0),
         #("Perceptron",0),
         #("KNeighbors",0),
         #("Tree",0),
         ("SGDClassifier",0),
         ("PassiveAggressive",0),
         ("LinearSVC",0),
         #("LSHForest",0)
         #("SVC",0)
         ("LogisticRegression",0),
         #("BaggingSVC",0),
         ("BaggingPerceptron",0),
         #("BaggingSGD",0)
         #("AdaBoost"0,)
         ]


#############################
#meta classifiers that combines them all


metaClassifiers = [ BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0)
                    #AdaBoostClassifier(n_estimators = 100,algorithm='SAMME.R'),
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto'),
                    #RandomForestClassifier(n_estimators=50),
                    #BaggingClassifier(SVC(),n_estimators = 10,max_samples=0.1, max_features=.5)
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]

metaNames = [("Bagging",0)
             #("AdaBoost",0),
             #("SVC",0)
             #("DecisionTreeGini",0),
             #("DecisionTreeEntropy",0),
             #("RandomForest",0),
             #("SVC",0)
             #("LinearSVC",0)
             #("LogisticRegression",0)
             ]

metaClassifiersWD = [ #BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),
                    AdaBoostClassifier(n_estimators = 250,algorithm='SAMME.R')
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto')
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]
metaNamesWD = [#("Bagging",0),
             ("AdaBoost",0)
             #("SVC",0)
             #("DecisionTreeGini",0),
             #("DecisionTreeEntropy",0)
             #("LinearSVC",0)
             #("Log",0)
             ]

#TRYING TO USE WORD2VEC....
#print "USING WORD2VEC TO PARSE"
#model = Word2Vec(trainingCorpus)
#model.init_sims(replace=True)
#model_name = "model1"
#model.save(model_name)
#model = Word2Vec.load('model1')
#trainingX = model.syn0




print "Extracting features based on the training set"
normalizedCountVectorizer = TfidfVectorizer(ngram_range=GRAM,stop_words='english',max_features=KBESTNUM)
testingExamples = normalizedCountVectorizer.fit_transform(testingExamples)
trainingX = normalizedCountVectorizer.transform(trainingCorpus)
print "NUM FEATURES = ", trainingX.shape[1]

############################################
#CHOOSE FEATURES
print "removing features with zero variance"
sel = VarianceThreshold()
trainingX = sel.fit_transform(trainingX)
testingExamples = sel.transform(testingExamples)
##print "feature selecting using " +str(KBESTNUM) 
##
##if KBESTNUM  > trainingX.shape[1]:
##    KBESTNUM = trainingX.shape[1]
###kBest        = SelectKBest(f_classif,k=KBESTNUM)
##kBest        = SelectKBest(chi2,k=KBESTNUM)
##trainingX       = kBest.fit_transform(trainingX,trainingY)
##testingExamples        = kBest.transform(testingExamples)

print "done selecting features"

metaTrainingSet = []
metaPredictionSet = []
############################################
#STRATIFIED K FOLD REGULAR CLASSIFIERS
if not DOING_PREDICTIONS and (RUN_REGS or RUN_METAS):
    print "Stratified K-Fold with " + str(KFOLD) + " folds"  
       
    Xtrain = trainingX
    Ytrain = trainingY
    
    counter = 0
    for (name,curScore),clf in zip(names,classifiers):
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        #predictionProbs = cross_val_predict_proba(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        names[counter] = (name,score)
        #ADD PREDICTIONS TO META TRAINING SET
        metaTrainingSet.append(predictions)
        counter += 1
        
    for name,score in names: print name ,float(score)#/float(KFOLD)


#METAS!!!!###
############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH JUST PREDICTIONS
if not DOING_PREDICTIONS and RUN_METAS:
    #print "Stratified K-Fold for METAS with " + str(KFOLD) + " folds"  
       
    Xtrain = np.array(metaTrainingSet).T
    Ytrain = trainingY
    
    counter = 0
    for (name,curScore),clf in zip(metaNames,metaClassifiers):
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        metaNames[counter] = (name,score)
        counter += 1
        
    for name,score in metaNames: print name ,float(score)

#METAS!!!!###
############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH DATA + PREDICTIONS (WD)
if not DOING_PREDICTIONS and RUN_METAS and RUN_WD:
    #CHOOSE FEATURES
    trainingXWD = trainingX
    #print "removing features with zero variance"
    sel = VarianceThreshold()
    trainingX = sel.fit_transform(trainingXWD)
    #print "feature selecting using " +str(KBESTNUMWD) 

    if KBESTNUMWD  > trainingXWD.shape[1]:
        KBESTNUMWD = trainingXWD.shape[1]
    #kBest        = SelectKBest(f_classif,k=KBESTNUM)
    kBest        = SelectKBest(chi2,k=KBESTNUMWD)
    trainingXWD       = kBest.fit_transform(trainingXWD,trainingY)

    #print "done selecting features"
    #print "Stratified K-Fold for METAWDs with " + str(KFOLD) + " folds"  

    metaArray = np.array(metaTrainingSet).T
    
    Xtrain = sparse.hstack((trainingXWD,metaArray))
    Ytrain = trainingY

    #print "done combining data and predictors"
    #print Xtrain.shape
    counter = 0
    for (name,curScore),clf in zip(metaNamesWD,metaClassifiersWD):
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        metaNamesWD[counter] = (name,score)
        counter += 1
        
    for name,score in metaNamesWD: print name + "DATA" ,float(score)

#######################################
##predict actual results!
if DOING_PREDICTIONS:
    for (name,curScore),clf in zip(names,classifiers):
        print "Prediction using " + name + " classifier"
        clf.fit(trainingX,trainingY)
        validationPredictions = clf.predict(testingExamples)
        metaPredictionSet.append(validationPredictions)
        with open('./Predictions/'+ name+ 'predictions.csv','wb') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 0
            for prediction in validationPredictions:
                writer.writerow([counter,prediction])
                counter += 1
    #METAS
    for (name,curScore),clf in zip(names,classifiers):
        print  name + " Training for metas"
        predictions = cross_val_predict(clf,trainingX,y=trainingY,cv=KFOLD)
        print name, accuracy_score(trainingY,predictions)
        #ADD PREDICTIONS TO META TRAINING SET
        metaTrainingSet.append(predictions)
        
    metaTrainingX   = np.array(metaTrainingSet).T
    metaPredictionX = np.array(metaPredictionSet).T
    
    for (name,curScore),clf in zip(metaNames,metaClassifiers):
        print "Prediction using " + "meta" + name + " classifier"
        clf.fit(metaTrainingX,trainingY)
        validationPredictions = clf.predict(metaPredictionX)
        with open('./Predictions/'+ "meta" + name+ 'predictions.csv','wb') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 0
            for prediction in validationPredictions:
                writer.writerow([counter,prediction])
                counter += 1
    if RUN_WD:
        #CHOOSE FEATURES
        trainingXWD = trainingX
        testingExamplesWD = testingExamples
        #print "removing features with zero variance"
        sel = VarianceThreshold()
        trainingXWD = sel.fit_transform(trainingXWD)
        testingExamplesWD = sel.transform(testingExamplesWD)
        #print "feature selecting using " +str(KBESTNUMWD) 

        if KBESTNUMWD  > trainingXWD.shape[1]:
            KBESTNUMWD = trainingXWD.shape[1]
        #kBest        = SelectKBest(f_classif,k=KBESTNUM)
        kBest        = SelectKBest(chi2,k=KBESTNUMWD)
        trainingXWD       = kBest.fit_transform(trainingXWD,trainingY)
        testingExamplesWD = kBest.transform(testingExamplesWD)
        metaTrainingXWD = sparse.hstack((trainingXWD,metaTrainingX))
        metaPredictionXWD = sparse.hstack((testingExamplesWD,metaPredictionX))
        for (name,curScore),clf in zip(metaNamesWD,metaClassifiersWD):
            print "Prediction using " + "meta" + name + " classifier"
            clf.fit(metaTrainingXWD,trainingY)
            validationPredictions = clf.predict(metaPredictionXWD)
            with open('./Predictions/'+ "metaWD" + name+ 'predictions.csv','wb') as predictions:
                writer = csv.writer(predictions)
                writer.writerow(['Id','Prediction'])
                counter = 0
                for prediction in validationPredictions:
                    writer.writerow([counter,prediction])
                    counter += 1
    

