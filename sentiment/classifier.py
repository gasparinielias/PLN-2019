from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, fit_grid_point
from sklearn.metrics import accuracy_score

from sentiment import tokenizer, preprocessor
from sentiment.settings import GRID_PARAMS
from sentiment.tweet2vec import Tweet2Vec


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


def score(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


class SentimentClassifier():

    def __init__(self, fasttext_model, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        tokenize = tokenizer.Tokenizer(
            max_negations=1,
            filter_stopwords=True
        )
        preprocess = preprocessor.Preprocessor()
        self._clf = clf

        self._pipeline = Pipeline([
            ('vect', FeatureUnion([
                ('count_vec', CountVectorizer(
                    preprocessor=preprocess,
                    tokenizer=tokenize,
                    binary=True,
                )),
                ('t2v', Tweet2Vec(fasttext_model)),
            ])),
            ('clf', classifiers[self._clf]()),
        ])

    def fit(self, X, y, train, test, grid_search=True):
        param_grid = ParameterGrid(GRID_PARAMS[self._clf]) if grid_search else []
        best_acc = -1
        best_params = {}
        for params in param_grid:
            accuracy, _, _ = fit_grid_point(X, y, self._pipeline, params,
                                            train, test, score, 0)
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = params

        # Refit using best params
        X = [X[i] for i in train]
        y = [y[i] for i in train]
        self._pipeline.set_params(**best_params)
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)

    def predict_proba(self, X):
        return self._pipeline.predict_proba(X)
