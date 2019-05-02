import csv
import re

from sentiment.consts import LEXICON_PATH


class PolarizedWordsCounter():
    def __init__(self, thresholds=(1.66, 2.33)):
        self._thresh = thresholds
        self._load_polarities()
        self.token_pattern = r"(?u)\b\w\w+\b"

    def fit(self, X, y):
        return self

    def build_tokenizer(self):
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def transform(self, X):
        X_t = []  # p, neu, n
        tokenize = self.build_tokenizer()
        for sent in X:
            classified_tokens = self.classify(tokenize(sent))
            pol_count = [
                len(classified_tokens['neg']),
                len(classified_tokens['neu']),
                len(classified_tokens['pos'])]

            X_t.append(pol_count)
        return X_t

    def classify(self, tokens):
        per_polarity = {'neg': [], 'pos': [], 'neu': []}
        for token in tokens:
            token_l = token.lower()
            if token_l in self.word_polarities:
                score = self.word_polarities.get(token_l, -1)
                if score == -1:
                    continue
                if score < self._thresh[0]:
                    per_polarity['neg'].append(token_l)
                elif score < self._thresh[1]:
                    per_polarity['neu'].append(token_l)
                else:
                    per_polarity['pos'].append(token_l)
        return per_polarity

    def set_params(self, **params):
        if 'thresh' in params:
            self._thresh = params['thresh']

    def get_feature_names(self):
        return ['neg', 'neu', 'pos']

    def _load_polarities(self):
        self.word_polarities = word_polarities = {}
        with open(LEXICON_PATH, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                tok, _ = row[0].split('_')
                pol = float(row[1])
                word_polarities[tok] = pol
