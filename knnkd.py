import pickle
import numpy as np
from scipy.sparse import *
import math
import csv
from scipy.sparse.linalg import norm
from multiprocessing import Process
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import Bootstrap
from scipy.spatial import KDTree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#returns the precision and recall specific to a certain class cl
def getPrecRec (cl,testY,prediction):
	testY = testY == cl
	prediction = prediction == cl
	truePos = sum(testY*prediction)
	a = [float(truePos)/float(sum(prediction)),float(truePos)/float(sum(testY))]
	return a
	
# returns a vector containing the predictions	
def getPred (index,trainY,useGau,dist,sigma):
	values = trainY[index]
	pred = []
	for i in range(0,4):
		curr = values == i
		if useGau: # if useGau is true we use gaussian weights, otherwise we use weights of 1 for all votes
			dist = np.exp((-1.0/sigma)*(dist**2))
			curr = curr*dist
		curr = np.sum(curr,axis=1)
		pred = np.append(pred,curr)
	print index.shape
	print pred.shape
	pred = pred.reshape((4,index.shape[0]))
	print pred.shape
	return np.argmax(pred,axis = 0)
				
def kNN (nb,trainX,trainY,crossX,s):
	tree = KDTree(trainX)
	(d,i) = tree.query(crossX,k=nb,p=2) # get nb nearest neighbours based on euclidean distance
	pred = getPred(i,trainY,False,d,s) 
	return pred

#returns the estimate of our score functions
def bootstrap(train_index,test_index,numnb,trainingX,trainingY,kBest,s) :
	trainX = trainingX[train_index]
	trainY = trainingY[train_index]
	testX = trainingX[test_index]
	testY = trainingY[test_index]
	trainX  = kBest.fit_transform(trainX,trainY) #select features using Kbest
	testX  = kBest.transform(testX)
	trainX = trainX.toarray()
	testX = testX.toarray()
	accuraccy = np.array([])
	prediction = kNN(numnb,trainX,trainY,testX,s)
	precision = np.array([])
	recall = np.array([])
	for cl in range(0,4):
		[a,b] = getPrecRec(cl,testY,prediction)
		precision = np.append(precision,a)
		recall = np.append(recall,b)
		accuraccy = np.append(accuraccy,float(sum((testY==cl) == (prediction==cl)))/float(testY.shape[0]))
	f1 = 2*precision*recall/(precision+recall)
	acc = float(sum(testY == prediction))/float(testY.shape[0])
	return [acc,accuraccy,precision,recall,f1]

if __name__ == '__main__':
	NUMFT = 30
	NUMIT = 5
	NUMNB = [2,4,7,10,15] #values for k
	sigma = [0.25,1.75,5,20] #values for sigma
	makePred = False #set to true if you want to make predictions
	getConM = True #set to true to get the confusion matrix using our optimal parameters
	
	with open("./DataSetWithDictionarys/trainingSetX.txt","rb") as trainingFileX:
		trainingX = pickle.load(trainingFileX)

	with open("./DataSetWithDictionarys/trainingSetY.txt","rb") as trainingFileY:
		trainingY = pickle.load(trainingFileY)

	with open("./DataSetWithDictionarys/validationSet.txt","rb") as validationFile:
		validationSet = pickle.load(validationFile)
		
	bs = Bootstrap(trainingX.shape[0],n_iter=NUMIT,random_state=0)
	kBest = SelectKBest(chi2,k=NUMFT)
	if makePred:
		for nb in NUMNB:
			for s in sigma:
				acc = np.array([])
				acpcl = np.array([0.0,0.0,0.0,0.0])
				pre = np.array([0.0,0.0,0.0,0.0])
				rec = np.array([0.0,0.0,0.0,0.0])
				f1score = np.array([0.0,0.0,0.0,0.0])
		
				for train_index, test_index in bs:
					[accuraccy,accpcl,precision,recall,f1] = (bootstrap(train_index,test_index,nb,trainingX,trainingY,kBest,s))
					acc = np.append(acc,accuraccy)
					acpcl = accpcl+acpcl
					pre = pre+precision
					rec = rec+recall
					f1score = f1score+f1
			
				acpcl = acpcl/float(NUMIT)
				accuraccy = sum(acc)/float(NUMIT)
				pre = pre/float(NUMIT)
				rec = rec/float(NUMIT)
				f1score = f1score/float(NUMIT)
				avpre = sum(pre)/4
				avrec = sum(rec)/4
				avf1 = sum(f1score/4)
				
				with open('./Predictions/KNN/choosenumfeat+.csv','a') as nbfeat:
					writer = csv.writer(nbfeat)
					writer.writerow([s,nb,accuraccy,acpcl[0],acpcl[1],acpcl[2],acpcl[3],pre[0],pre[1],pre[2],pre[3],rec[0],rec[1],rec[2],rec[3],f1score[0],f1score[0],f1score[0],f1score[0],avpre,avrec,avf1])
		
	if getConM:
		for train_index, test_index in bs:
			trainX = trainingX[train_index]
			print trainX.shape
			trainY = trainingY[train_index]
			testX = trainingX[test_index]
			testY = trainingY[test_index]
			trainX  = kBest.fit_transform(trainX,trainY)
			testX  = kBest.transform(testX)
			trainX = trainX.toarray()
			testX = testX.toarray()
			predictions = kNN(15,trainX,trainY,testX,1.75)
			plt.figure()
			cm = confusion_matrix(testY, predictions)
			plt.xlabel('Predicted class')
			plt.ylabel('Actual class')
			plt.title('KNN Confusion Matrix')
			plt.imshow(cm,interpolation='nearest')
			plt.savefig('./Figures/CM/KNN.png')
			break