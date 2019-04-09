from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class neg_handling_tokenizer():
    def __init__(self, max_negations=0, filter_stopwords=True):
        self.max_negations = max_negations
        self.filter_stopwords = filter_stopwords

    def __call__(self, s):
        tokens = word_tokenize(s)

        if self.filter_stopwords:
            spanish_sw = set(stopwords.words('spanish'))
            tokens = [token for token in tokens if token not in spanish_sw or token in ['no', 'ni']]

        neg_words = ['no', 'tampoco', 'ni']
        neg_count = 0
        make_opposite = False
        if self.max_negations > 0:
            for i in range(len(tokens)):
                if re.search(r'[.?\-":!,]+', tokens[i]):
                    # punctuation mark found
                    make_opposite = False
                if make_opposite and \
                    tokens[i] not in neg_words and \
                    neg_count < self.max_negations:

                    tokens[i] = 'NOT_' + tokens[i]
                    neg_count += 1
                if tokens[i] in neg_words:
                    make_opposite = True
                    neg_count = 0
        return tokens
