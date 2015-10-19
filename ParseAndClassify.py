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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pylab
#FLAGS#
DOING_PREDICTIONS = False
RUN_METAS         = True
RUN_REGS          = True
RUN_WD            = True
#K best features, (1,3) gram = 30k,(1,2) = 25k,(1,1)=7.5k
KBESTNUM = 500000
KBESTNUMWD = 10
#K FOLD
KFOLD  = 10

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
#Classifier list, to be iterated over (name,score,classifier,truePositiveRate,falsePositiveRate,roc_auc)
classifiers = [
    #("MultinomialNB",0,MultinomialNB(),dict(),dict(),dict()),
    #("Perceptron",0,Perceptron(),dict(),dict(),dict()),
    #("KNeighbors",0,KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',leaf_size=1),dict(),dict(),dict()),
    #("Tree",0,DecisionTreeClassifier(max_depth=500,max_features=.05),dict(),dict(),dict()),
    #("SGDClassifier",0,SGDClassifier(),dict(),dict(),dict()),
    ("PassiveAggressive",0,PassiveAggressiveClassifier(),dict(),dict(),dict()),
    ("LinearSVC",0,LinearSVC(),dict(),dict(),dict()),
    #("LSHForest",0,LSHForest(),dict(),dict(),dict()),
    #("BaggingSVC",0,BaggingClassifier(SVC(cache_size = 500,degree = 2),n_estimators = 100,max_samples=0.001, max_features=0.5),dict(),dict(),dict()),
    #("LogisticRegression",0,LogisticRegression(max_iter = 100, solver = 'newton-cg'),dict(),dict(),dict()),
    #BaggingClassifier(LinearSVC(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict()),
    #("BaggingPerceptron",0,BaggingClassifier(Perceptron(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict())
    #("BaggingSGD",0,BaggingClassifier(SGDClassifier(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict()),
    #("AdaBoost"0,AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 1000,algorithm='SAMME.R'),dict(),dict(),dict())
    ]



#############################
#meta classifiers that combines them all


metaClassifiers = [ ("Bagging",0,BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),dict(),dict(),dict())
                    #AdaBoostClassifier(n_estimators = 100,algorithm='SAMME.R'),
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto'),
                    #RandomForestClassifier(n_estimators=50),
                    #BaggingClassifier(SVC(),n_estimators = 10,max_samples=0.1, max_features=.5)
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]



metaClassifiersWD = [ #BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),
                    ("AdaBoost",0,AdaBoostClassifier(n_estimators = 250,algorithm='SAMME.R'),dict(),dict(),dict())
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto')
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
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
    for (name,curScore,clf,tpr,fpr,aucRoc) in classifiers:
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        #predictionProbs = cross_val_predict_proba(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        print name,score
        #ROC CURVE
        for i in range(0,4):
            fpr[i], tpr[i], threshold = roc_curve(Ytrain, predictions, pos_label=i)
            aucRoc[i] = auc(fpr[i],tpr[i])
        #ADD scores and curves
        classifiers[counter] = (name,score,clf,tpr,fpr,aucRoc)
        #ADD PREDICTIONS TO META TRAINING SET
        metaTrainingSet.append(predictions)
        counter += 1
        
    


#METAS!!!!###
############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH JUST PREDICTIONS
if not DOING_PREDICTIONS and RUN_METAS:
    #print "Stratified K-Fold for METAS with " + str(KFOLD) + " folds"  
       
    Xtrain = np.array(metaTrainingSet).T
    Ytrain = trainingY
    
    counter = 0
    for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiers:
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        print name,score
        #ROC CURVE
        for i in range(0,4):
            fpr[i], tpr[i], threshold = roc_curve(Ytrain, predictions, pos_label=i)
            aucRoc[i] = auc(fpr[i],tpr[i])
        #ADD scores and curves
        metaClassifiers[counter] = (name,score,clf,tpr,fpr,aucRoc)
        counter += 1
        
    #for name,score in metaNames: print name ,float(score)

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
    for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiersWD:
        print  name + " Training"
        predictions = cross_val_predict(clf,Xtrain,y=Ytrain,cv=KFOLD)
        score = accuracy_score(Ytrain,predictions)
        print name,score
        #ROC CURVE
        for i in range(0,4):
            fpr[i], tpr[i], threshold = roc_curve(Ytrain, predictions, pos_label=i)
            aucRoc[i] = auc(fpr[i],tpr[i])
        #ADD scores and curves
        metaClassifiers[counter] = (name,score,clf,tpr,fpr,aucRoc)
        counter += 1
        
    #for name,score in metaNamesWD: print name + "DATA" ,float(score)

#OUTPUT GRAPHS AND SCORES
if not DOING_PREDICTIONS:
    for i in range(0,4):
        plt.figure()
        if RUN_REGS:
            for (name,curScore,clf,tpr,fpr,aucRoc) in classifiers:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label= name+ '(area = %0.2f)' % aucRoc[i])
        if RUN_METAS:
            for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiers:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label='meta '+name+ '(area = %0.2f)' % aucRoc[i])
        if RUN_WD:
            for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiersWD:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label='metaWD '+name+ '(area = %0.2f)' % aucRoc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic' +str(i))
        plt.legend(loc="lower right")
        pylab.savefig('./Figures/Roc/'+'ROC'+str(i))


#######################################
##predict actual results!
if DOING_PREDICTIONS:
    for (name,curScore,clf,tpr,fpr,auc) in classifiers:
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
    for (name,curScore,clf,tpr,fpr,auc) in classifiers:
        print  name + " Training for metas"
        predictions = cross_val_predict(clf,trainingX,y=trainingY,cv=KFOLD)
        print name, accuracy_score(trainingY,predictions)
        #ADD PREDICTIONS TO META TRAINING SET
        metaTrainingSet.append(predictions)
        
    metaTrainingX   = np.array(metaTrainingSet).T
    metaPredictionX = np.array(metaPredictionSet).T
    
    for (name,curScore,clf,tpr,fpr,auc) in metaClassifiers:
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
        for (name,curScore,clf,tpr,fpr,auc) in metaClassifiersWD:
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
    

