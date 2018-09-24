from Featurizer import Featurizer
from sklearn.feature_extraction.text import CountVectorizer

#This is a subclass that extends the abstract class Featurizer.
class CountFeaturizer(Featurizer):
    def __init__(self, stop_words):
        self.count_vect = CountVectorizer(stop_words=stop_words)

    #The abstract method from the base class is implemeted here to return count features
    def getFeatureRepresentation(self, X_train, X_val):

        X_train_counts = self.count_vect.fit_transform(X_train)
        X_val_counts = self.count_vect.transform(X_val)

        return X_train_counts, X_val_counts