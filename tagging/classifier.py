from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from tagging.consts import START_TOKEN, END_TOKEN
from tagging.fasttext import FasttextDictVectorizer


classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
    'nb': MultinomialNB
}


def add_keys(d, word, prefix):
    if word == START_TOKEN or word == END_TOKEN:
        fun = (str.lower,)
    else:
        fun = (str.lower, str.isupper, str.istitle, str.isdigit)

    keys = [
        (f, "{}{}".format(prefix, key))
        for f, key in zip(fun, 'w wu wt wd'.split())
    ]

    d.update({ key: f(word) for f, key in keys })

def feature_dict(sent, i):
    """Feature dictionary for a given sentence and position.

    sent -- the sentence.
    i -- the position.
    """
    assert 0 <= i and i < len(sent)

    cword = sent[i]
    pword = sent[i - 1] if i != 0 else START_TOKEN
    nword = sent[i + 1] if i != len(sent) - 1 else END_TOKEN

    fdict = {}
    add_keys(fdict, cword, '')
    add_keys(fdict, pword, 'p')
    add_keys(fdict, nword, 'n')

    return fdict


class ClassifierTagger():
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        self._clf = clf
        self._pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', classifiers[clf]())
        ])
        self.fit(tagged_sents)

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        X, y, vocab = self.prepare_fit(tagged_sents)

        self._vocab = vocab
        self._pipeline.fit(X, y)

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        X = self.prepare_predict(sents)
        return self._pipeline.predict(X)

    def prepare_fit(self, tagged_sents):
        X, y = [], []
        vocab = set()
        for sent in tagged_sents:
            if not len(sent):
                continue
            words, tags = zip(*sent)
            for i in range(len(words)):
                X.append(feature_dict(words, i))
            y.extend(tags)
            vocab.update(words)
        return X, y, vocab

    def prepare_predict(self, sents):
        X = []
        for sent in sents:
            for i in range(len(sent)):
                X.append(feature_dict(sent, i))
        return X
        
    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        X = self.prepare_predict([sent])
        return self._pipeline.predict(X)

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._vocab


class EmbeddingsTagger(ClassifierTagger):

    def __init__(self, tagged_sents, clf='lr'):
        self._clf = clf
        self._pipeline = Pipeline([
            ('vect', FeatureUnion([
                ('twv', DictVectorizer()),
                ('ft', FasttextDictVectorizer('models/cc.es.300.bin', ['w']))
            ])),
            ('clf', classifiers[clf]())
        ])
        self.fit(tagged_sents)
