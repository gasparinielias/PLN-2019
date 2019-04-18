from nltk.tokenize import word_tokenize
import numpy as np

class Tweet2Vec():
    def __init__(self, fasttext_model):
        self._model = fasttext_model

    def fit(self, X, y):
        return self

    def transform(self, tweets):
        wv = self._model.wv
        res = []
        for tweet in tweets:
            vecs = []
            for word in word_tokenize(tweet):
                try:
                    vecs.append(wv.word_vec(word))
                except KeyError:
                    pass

            res.append(np.average(vecs, axis=0))
        return res
