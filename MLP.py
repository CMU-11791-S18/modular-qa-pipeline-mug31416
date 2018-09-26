from Classifier import Classifier
from sklearn.neural_network import MLPClassifier


class MLP(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = MLPClassifier(hidden_layer_sizes=(100,100)).fit(X_features, Y_train)
        return clf

    def getName(self):
        return "MLP"