import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, fit_grid_point
from sklearn.exceptions import NotFittedError

from sentiment.tokenizer import neg_handling_tokenizer
from sentiment.preprocessor import preprocessor
from sentiment.settings import grid_params


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}

def score(estimator, X, y):
    return np.average(estimator.predict(X) == y)

class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        tokenize = neg_handling_tokenizer(
            max_negations=1,
            filter_stopwords=True
        )
        preprocess = preprocessor()
        self._clf = clf

        self._pipeline = Pipeline([
            ('vect', CountVectorizer(
                preprocessor=preprocess,
                tokenizer=tokenize,
                binary=True,
            )),
            ('clf', classifiers[self._clf]()),
        ])

    def fit(self, X, y, train, test):
        param_grid = ParameterGrid(grid_params[self._clf])
        best_acc = -1
        best_params = None
        for params in param_grid:
            accuracy, _, _ =  fit_grid_point(X, y, self._pipeline, params, train, test, score, 0)
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = params
                print(best_acc, best_params)

        X = [X[i] for i in train]
        y = [y[i] for i in train]
        self._pipeline.set_params(**best_params)
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
