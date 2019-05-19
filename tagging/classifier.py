from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from tagging.consts import START_TAG, END_TAG


classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
}


def add_keys(d, word, prefix):
    if word == START_TAG or word == END_TAG:
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
    pword = sent[i - 1] if i != 0 else START_TAG
    nword = sent[i + 1] if i != len(sent) - 1 else END_TAG

    fdict = {}
    add_keys(fdict, cword, '')
    add_keys(fdict, pword, 'p')
    add_keys(fdict, nword, 'n')

    return fdict


class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        self._clf = clf

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        # WORK HERE!!

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        # WORK HERE!!

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        # WORK HERE!!

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        # WORK HERE!!
