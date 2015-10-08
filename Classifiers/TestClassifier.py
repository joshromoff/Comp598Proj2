from AbstractClassifier import AbstractClassifier

class TestClassifier(AbstractClassifier):
    def __init__(self,training,test):
        self.trainingSet = training
        self.testSet     = test

    def Classify(self,text):
        return 1
