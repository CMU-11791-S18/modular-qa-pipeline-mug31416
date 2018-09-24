from Classifier import Classifier
from sklearn.svm import SVC


#This is a subclass that extends the abstract class Classifier.
class SVM(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = SVC(kernel='linear',random_state=0,probability=True).fit(X_features, Y_train)
        return clf