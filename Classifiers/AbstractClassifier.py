import abc

#abstract class for classifiers
#each classifier needs some function Classify that...
#takes as input a tokenized string and outputs a class (1,2,3,4)

class AbstractClassifier(object):
    __metaclass__ = abc.ABCMeta

    #abstract method Classify that must be implemented
    def Classify(self,text):
        return
