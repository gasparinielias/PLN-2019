from collections import defaultdict
import random

from languagemodeling.consts import START_TOKEN, END_TOKEN


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)

        for ngram in model._count.keys():
            if len(ngram) == self._n:
               probs[ngram[:-1]][ngram[-1]] = model.cond_prob(ngram[-1], ngram[:-1])
        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = {}
        for ngram, probs in self._probs.items():
            sorted_probs[ngram] = []
            for key, value in sorted(probs.items(), key=lambda item: (-item[1], item[0])):
                sorted_probs[ngram].append((key, value))
        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        sent = [START_TOKEN] * (self._n - 1)
        while END_TOKEN not in sent:
            u = random.random()
            last_ngram = tuple(sent[len(sent) - self._n + 1:])
            probs = self._sorted_probs[last_ngram]
            i = 0
            acum_prob = probs[i][1]
            while acum_prob < u:
                i += 1
                acum_prob += probs[i][1]
            sent += [probs[i][0]]

        return sent[self._n - 1:-1]


    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        assert((prev_tokens is not None) or self._n == 1)

        if prev_tokens not in self._sorted_probs.keys():
            return ''

        u = random.random()
        i = 0
        probs = self._sorted_probs[prev_tokens]
        acum_prob = probs[i][1]
        while acum_prob < u:
            i += 1
            acum_prob += probs[i][1]
        return probs[i][0]
