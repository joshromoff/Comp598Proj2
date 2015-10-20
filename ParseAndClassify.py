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
from Classifiers.LinearProbSVC import LinearProbSVC
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
#CURENT DEVELOPMENT VERSION
from cross_validation import cross_val_apply
#from sklearn.cross_validation import cross_val_predict_proba
from sklearn.metrics import accuracy_score
#from gensim.models import Word2Vec
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
RUN_WDP           = True
#K best features, (1,3) gram = 30k,(1,2) = 25k,(1,1)=7.5k
KBESTNUM = 500000
KBESTNUMMETA = 10
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
#Classifier list, to be iterated over (name,score,classifier,truePositiveRate,falsePositiveRate,roc_auc)
classifiers = [
    ("MultinomialNB",0,MultinomialNB(),dict(),dict(),dict()),
    #("Perceptron",0,Perceptron(),dict(),dict(),dict()),
    #("KNeighbors",0,KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',leaf_size=1),dict(),dict(),dict()),
    #("Tree",0,DecisionTreeClassifier(max_depth=10,max_features=.2),dict(),dict(),dict()),
    #("SGDClassifier",0,SGDClassifier(),dict(),dict(),dict()),
    #("PassiveAggressive",0,PassiveAggressiveClassifier(),dict(),dict(),dict()),
    #("LinearProbSVC",0,LinearProbSVC(),dict(),dict(),dict()),
    #("LSHForest",0,LSHForest(),dict(),dict(),dict()),
    #("BaggingSVC",0,BaggingClassifier(SVC(cache_size = 500,degree = 2),n_estimators = 100,max_samples=0.001, max_features=0.5),dict(),dict(),dict()),
    #("LogisticRegression",0,LogisticRegression(max_iter = 100, solver = 'newton-cg'),dict(),dict(),dict()),
    #("SVC",0,SVC(kernel='linear',degree=1,max_iter =25,probability =True),dict(),dict(),dict())
    #("BaggingLSVC",0,BaggingClassifier(LinearSVC(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict()),
    #("BaggingPerceptron",0,BaggingClassifier(Perceptron(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict())
    #("BaggingSGD",0,BaggingClassifier(SGDClassifier(),n_estimators = 100,max_samples=0.5, max_features=0.5),dict(),dict(),dict()),
    #("AdaBoost"0,AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 1000,algorithm='SAMME.R'),dict(),dict(),dict())
    ]



#############################
#meta classifiers that combines them all


metaClassifiers = [ #("Bagging",0,BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),dict(),dict(),dict())
                    ("AdaBoostMETA",0,AdaBoostClassifier(n_estimators = 50,algorithm='SAMME.R'),dict(),dict(),dict())
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto'),
                    #RandomForestClassifier(n_estimators=50),
                    #BaggingClassifier(SVC(),n_estimators = 10,max_samples=0.1, max_features=.5)
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]



metaClassifiersWD = [ #BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),
                    ("AdaBoostWD",0,AdaBoostClassifier(n_estimators = 50,algorithm='SAMME.R'),dict(),dict(),dict())
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto')
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]

metaClassifiersWDP = [ #BaggingClassifier(n_estimators = 250,max_samples=0.5, max_features=1.0),
                    ("AdaBoostWDP",0,AdaBoostClassifier(n_estimators = 50,algorithm='SAMME.R'),dict(),dict(),dict())
                    #SVC()
                    #DecisionTreeClassifier(criterion='gini',class_weight ='auto'),
                    #DecisionTreeClassifier(criterion='entropy',class_weight ='auto')
                    #LinearSVC()
                    #LogisticRegression(max_iter = 100, solver = 'newton-cg')
                    ]


#WRITE PREDICTIONS TO FILE
def predict_writeToFile(classList,x_train,y_train,x_test):
    for (name,curScore,clf,tpr,fpr,auc) in classList:
            print "Prediction using " + name + " classifier"
            clf.fit(x_train,y_train)
            validationPredictions = clf.predict(x_test)
            with open('./Predictions/'+ name+ 'predictions.csv','wb') as predictions:
                writer = csv.writer(predictions)
                writer.writerow(['Id','Prediction'])
                counter = 0
                for prediction in validationPredictions:
                    writer.writerow([counter,prediction])
                    counter += 1

#########################################
#PREDICT AND RETURN PREDICTIONS AND PREDICT PROBAS, SAVE CONFUSION MATRIX
def predict_Training(classList,x_train,y_train,kfold):
    predictProbsSet = []
    predictSet = []
    counter = 0
    for (name,curScore,clf,tpr,fpr,aucRoc) in classList:
        print  name + " Training"
        #GET PREDICTION PROBABILITIES
        predictProbs = cross_val_apply(clf,x_train,y=y_train,cv=kfold,apply_func ="predict_proba")
        #PICK HIGHEST PROB AS PREDICTION
        predictions  = [row.argmax() for row in predictProbs]
        #APPEND TO SET
        predictSet.append(predictions)
        #SCORE
        score = accuracy_score(y_train,predictions)
        print name,score

        #APPEND TO PROBABILITY SET
        for i in range(0,4):
            predictProbsSet.append(predictProbs[:,i])

        #GRAPHS ROC/CONFUSION IF NOT DOING PREDICTIONS
        if not DOING_PREDICTIONS:
            #ROC CURVE
            for i in range(0,4):
                fpr[i], tpr[i], threshold = roc_curve(y_train, predictProbs[:,i], pos_label=i)
                aucRoc[i] = auc(fpr[i],tpr[i])
                
            #ADD scores and curves
            classList[counter] = (name,score,clf,tpr,fpr,aucRoc)
            #CONFUSION MATRIX
            plt.figure()
            cm = confusion_matrix(y_train, predictions)
            plt.imshow(cm,interpolation='nearest')
            plt.savefig('./Figures/CM/'+name+'cm.png')
        
        counter +=1
        
    return predictSet,predictProbsSet

#####################################
#PLOT ROC CURVES
def plot_ROC():
    for i in range(0,4):
        plt.figure()
        if RUN_REGS:
            for (name,curScore,clf,tpr,fpr,aucRoc) in classifiers:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label= name + '(area = %0.2f)' % aucRoc[i])
        if RUN_METAS:
            for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiers:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label=name+ '(area = %0.2f)' % aucRoc[i])
        if RUN_WD:
            for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiersWD:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label=name+ '(area = %0.2f)' % aucRoc[i])
        if RUN_WDP:
            for (name,curScore,clf,tpr,fpr,aucRoc) in metaClassifiersWDP:
            # Plot of a ROC curve for a specific class
                plt.plot(fpr[i], tpr[i],label=name+ '(area = %0.2f)' % aucRoc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic' +str(i))
        plt.legend(loc="lower right")
        plt.savefig('./Figures/Roc/'+'ROC'+str(i))

############################################
#CHOOSE FEATURES, REMOVE ZERO VARIANCE AND SELECT KBEST BASED ON CHI2
def chooseFeatures(train_x,train_y,test_x,kB):
    
    sel = VarianceThreshold()
    trainingX = sel.fit_transform(train_x)
    testingExamples = sel.transform(test_x)
    

    if kB  > trainingX.shape[1]:
        kB = trainingX.shape[1]
    
    kBest        = SelectKBest(chi2,k=kB)
    train_x       = kBest.fit_transform(train_x,train_y)
    test_x        = kBest.transform(test_x)

    return train_x,test_x

#VECTORIZE
print "Extracting features based on the training set"
normalizedCountVectorizer = TfidfVectorizer(ngram_range=GRAM,stop_words='english',max_features=KBESTNUM)
testingExamples = normalizedCountVectorizer.fit_transform(testingExamples)
trainingX = normalizedCountVectorizer.transform(trainingCorpus)
print "NUM FEATURES = ", trainingX.shape[1]

#INITIAL FEATURE SELECTION
print "feature selecting using " +str(KBESTNUM)
trainingX,testingExamples = chooseFeatures(trainingX,trainingY,testingExamples,KBESTNUM)
print "done selecting features"

#GLOBAL META SETS
metaTrainingSet = []
metaTrainingSetProbs = []
metaPredictionSet = []
metaPredictionSetProbs = []

############################################
#STRATIFIED K FOLD REGULAR CLASSIFIERS
if not DOING_PREDICTIONS and RUN_REGS:
    print "Stratified K-Fold with " + str(KFOLD) + " folds"  
    #PREDICT AND SET APPEND PREDICTIONS AND PREDICTION PROBS FOR METAS
    metaTrainingSet, metaTrainingSetProbs = predict_Training(classifiers,trainingX,trainingY,KFOLD)
    


#METAS!!!!###
############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH JUST PREDICTIONS
if not DOING_PREDICTIONS and RUN_METAS:
    #LOCAL TRAINING VARIABLE
    Xtrain = np.array(metaTrainingSet).T

    #PREDICT
    predict_Training(metaClassifiers,Xtrain,trainingY,KFOLD)
    

#METAS!!!!###
############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH DATA + PREDICTIONS (WD)
if not DOING_PREDICTIONS and RUN_WD:
    #LOCAL TRAINING VARIABLE
    trainingXWD = trainingX

    #CHOOSE FEATURES
    trainingXWD,_ = chooseFeatures(trainingXWD,trainingY,trainingXWD,KBESTNUMMETA)

    #TRANSFORM TO ARRAY AND APPEND PREDICTIONS TO KBESTMETA FEATURES
    metaArray = np.array(metaTrainingSet).T
    Xtrain = sparse.hstack((trainingXWD,metaArray))

    #PREDICT
    predict_Training(metaClassifiersWD,Xtrain,trainingY,KFOLD)

############################################
#STRATIFIED K FOLD META CLASSIFIERS WITH DATA + PREDICTIONS (WD)
if not DOING_PREDICTIONS and RUN_WDP:
    #LOCAL TRAINING VARIABLE
    trainingXWDP = trainingX
    
    #CHOOSE FEATURES
    trainingXWDP,_ = chooseFeatures(trainingXWDP,trainingY,trainingXWDP,KBESTNUMMETA)

    #TRANSFORM TO ARRAY AND APPEND PREDICTION PROBS TO KBESTMETA FEATURES
    metaArray = np.array(metaTrainingSetProbs).T
    Xtrain = sparse.hstack((trainingXWDP,metaArray))

    #PREDICT
    predict_Training(metaClassifiersWDP,Xtrain,trainingY,KFOLD)

      
#OUTPUT GRAPHS AND SCORES
if not DOING_PREDICTIONS:
    plot_ROC()

    
#######################################
##OUTPUT RESULTS TO FILE
if DOING_PREDICTIONS:
    for (name,curScore,clf,tpr,fpr,auc) in classifiers:
        print "Prediction using " + name + " classifier"
        clf.fit(trainingX,trainingY)
        print trainingX.shape,testingExamples.shape
        validationProbs = clf.predict_proba(testingExamples)
        #validationPredictions = clf.predict(testingExamples)
        validationPredictions = [row.argmax() for row in validationProbs]
        metaPredictionSet.append(validationPredictions)
        for i in range(0,4):
            metaPredictionSetProbs.append(validationProbs[:,i])
        with open('./Predictions/'+ name+ 'predictions.csv','wb') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 0
            for prediction in validationPredictions:
                writer.writerow([counter,prediction])
                counter += 1

    #CROSS VALIDATION PREDICTIONS FOR METAS            
    metaTrainingSet,metaTrainingSetProbs = predict_Training(classifiers,trainingX,trainingY,KFOLD)
    
    #META BASIC ARRAY TRANSFORMATIONS
    metaTrainingX   = np.array(metaTrainingSet).T
    metaPredictionX = np.array(metaPredictionSet).T
    metaTrainingXProbs   = np.array(metaTrainingSetProbs).T
    metaPredictionXProbs = np.array(metaPredictionSetProbs).T

    if RUN_METAS:
        predict_writeToFile(metaClassifiers,metaTrainingX,trainingY,metaPredictionX)

    if RUN_WD:
        #CREATE LOCAL VARS
        trainingXWD = trainingX
        testingExamplesWD = testingExamples

        #CHOOSE FEATURES
        trainingXWD,testingExamplesWD = chooseFeatures(trainingXWD,trainingY,testingExamplesWD,KBESTNUMMETA)

        #TRANSFORM FEATURES + BASE CLASSIFIER PREDICTIONS
        metaTrainingXWD = sparse.hstack((trainingXWD,metaTrainingX))
        metaPredictionXWD = sparse.hstack((testingExamplesWD,metaPredictionX))
        predict_writeToFile(metaClassifiersWD,metaTrainingXWD,trainingY,metaPredictionXWD)

    if RUN_WDP:
        #CREATE LOCAL VARS
        trainingXWDP = trainingX
        testingExamplesWDP = testingExamples
        
        #CHOOSE FEATURES
        trainingXWDP,testingExamplesWDP = chooseFeatures(trainingXWDP,trainingY,testingExamplesWDP,KBESTNUMMETA)

        #TRANSFORM FEATURES + BASE CLASSIFIER PREDICTIONS
        metaTrainingXWDP = sparse.hstack((trainingXWDP,metaTrainingXProbs))
        metaPredictionXWDP = sparse.hstack((testingExamplesWDP,metaPredictionXProbs))
        
        predict_writeToFile(metaClassifiersWDP,metaTrainingXWDP,trainingY,metaPredictionXWDP)

    

