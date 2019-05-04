import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Tokenizer():

    def __init__(self, max_negations=0, filter_stopwords=True):
        self.max_negations = max_negations
        self.filter_stopwords = filter_stopwords

    def __call__(self, sent):
        tokens = word_tokenize(sent)

        neg_words = ['no', 'tampoco', 'ni']

        if self.filter_stopwords:
            spanish_sw = set(stopwords.words('spanish'))
            tokens = list(filter(
                lambda tok: tok not in spanish_sw or tok in neg_words,
                tokens))

        neg_count = 0
        make_opposite = False
        if self.max_negations > 0:
            for i, token in enumerate(tokens):
                if re.search(r'[.?\-":!,]+', token):
                    # punctuation mark found
                    make_opposite = False

                if make_opposite and \
                   token not in neg_words and \
                   neg_count < self.max_negations:

                    tokens[i] = 'NOT_' + token
                    neg_count += 1

                if token in neg_words:
                    make_opposite = True
                    neg_count = 0

        return tokens
