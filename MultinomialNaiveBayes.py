from Classifier import Classifier
from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayes(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = MultinomialNB(fit_prior=False).fit(X_features, Y_train)
        return clf