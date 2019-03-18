from collections import defaultdict
import random


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
            sorted_probs[ngram] = {}
            for key, value in sorted(probs.items(), key=lambda item: (-item[1], item[0])):
                sorted_probs[ngram][key] = value
        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        sent = ['<s>'] * (self._n - 1)
        while '</s>' not in sent:
            u = random.random()
            last_ngram = tuple(sent[len(sent) - self._n + 1:])
            prob_dict = self._sorted_probs[last_ngram]
            acum_prob = 0
            for word, prob in prob_dict.items():
                acum_prob += prob
                if u < acum_prob:
                    sent += [word]
                    break

        return sent


    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
