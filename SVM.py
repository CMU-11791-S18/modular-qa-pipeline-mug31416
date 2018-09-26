from Classifier import Classifier
from sklearn.svm import SVC


class SVM(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = SVC(kernel='linear',random_state=0,probability=False).fit(X_features, Y_train)
        return clf

    def getName(self):
        return "SVM"