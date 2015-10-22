# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:21:37 2015

@author: Eric
"""

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
mat = [[3487,  736,  710,  842], [ 644, 1860,  579,  477], [ 512,  541, 3195,  463], [1225,  680,  576, 1044]]
plot_confusion_matrix(mat)