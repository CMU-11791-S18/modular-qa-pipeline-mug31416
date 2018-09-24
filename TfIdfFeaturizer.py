from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer
#This is a subclass that extends the abstract class Featurizer.

class TfIdfFeaturizer(Featurizer):
    def __init__(self, stop_words):
        self.tfidf_vect = TfidfVectorizer(stop_words=stop_words)

    #The abstract method from the base class is implemeted here to return count features
    def getFeatureRepresentation(self, X_train, X_val):
        X_train_counts = self.tfidf_vect.fit_transform(X_train)
        X_val_counts = self.tfidf_vect.transform(X_val)
        return X_train_counts, X_val_counts