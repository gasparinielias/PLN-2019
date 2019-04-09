from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sentiment.tokenizer import neg_handling_tokenizer
from sentiment.preprocessor import preprocessor

classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


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
        self._pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer(
                #preprocessor=preprocess,
                tokenizer=tokenize,
                binary=True
                )),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
