# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import numpy as np

from languagemodeling.consts import *


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """
        log_p = 0
        for sent in sents:
            log_p += self.sent_log_prob(sent)
        return log_p

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        wc = sum(map(len, sents)) + len(sents)  # each sent lacking of END_TOKEN
        ce = - self.log_prob(sents) / wc
        return ce

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        return 2 ** self.cross_entropy(sents)


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n
        self._addone = False

        self._compute_counts(sents)

    def _compute_counts(self, sents, all_ngrams=False):
        n = self._n
        self._count = count = defaultdict(int)

        min_ngram = 1 if all_ngrams else max(n - 1, 1)
        for i in range(min_ngram, n):
            count[('<s>',) * i] += len(sents)

        for sent in sents:
            count[()] += len(sent) + 1
            sent = [START_TOKEN] * (n - 1) + sent + [END_TOKEN]

            for i in range(min_ngram, n + 1):
                for j in range(n - i, len(sent) - i + 1):
                    count[tuple(sent[j:j + i])] += 1

        if self._addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            for sent in sents:
                for word in sent:
                    voc.add(word)
            voc.add(END_TOKEN)

            self._V = len(voc)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        assert (prev_tokens is not None) or self._n == 1

        prev_tokens = prev_tokens or ()

        if self.count(prev_tokens) == 0:
            return 0
        return self.count(prev_tokens + (token,)) / self.count(prev_tokens)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self._n
        p = 1
        sent = [START_TOKEN] * (n - 1) + sent + [END_TOKEN]
        for i in range(len(sent) - n + 1):
            token = sent[i + n - 1]
            prev_tokens = tuple(sent[i:i + n - 1])
            p *= self.cond_prob(token, prev_tokens)
        return p

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        n = self._n
        p = 0
        sent = [START_TOKEN] * (n - 1) + sent + [END_TOKEN]
        for i in range(len(sent) - n + 1):
            token = sent[i + n - 1]
            prev_tokens = tuple(sent[i:i + n - 1])
            if self.cond_prob(token, prev_tokens) == 0:
                return -math.inf
            p += math.log(self.cond_prob(token, prev_tokens), 2)
        return p


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0

        self._n = n
        self._addone = True
        self._compute_counts(sents)

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        assert (prev_tokens is not None) or self._n == 1

        prev_tokens = prev_tokens or ()

        return (self.count(prev_tokens + (token,)) + 1) / \
            (self.count(prev_tokens) + self._V)


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        self._addone = addone
        self._compute_counts(train_sents, all_ngrams=True)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        elif self._n > 1:
            # Otherwise parameter gamma is not needed
            print('Computing gamma...')
            # use grid search to choose gamma
            best_gamma = None
            best_perplexity = math.inf
            for gamma in np.linspace(GAMMA_MIN, GAMMA_MAX, GAMMA_LINSP_NUM):
                self._gamma = gamma
                perplexity = self.perplexity(held_out_sents)
                if best_gamma is None or perplexity < best_perplexity:
                    best_gamma = gamma
                    best_perplexity = perplexity
            self._gamma = best_gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._count[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        assert (prev_tokens is not None) or self._n == 1

        prev_tokens = prev_tokens or ()

        lambdas = self._compute_lambdas(prev_tokens)

        probs = []
        for i in range(self._n):
            if self.count(prev_tokens[i:]) != 0:
                probs.append(self.count(prev_tokens[i:] + (token,)) /
                             self.count(prev_tokens[i:]))
            else:
                probs.append(0)

        if self._addone:
            probs[-1] = (self.count((token,)) + 1) / (self.count(()) + self._V)

        return sum(l * p for l, p in zip(lambdas, probs))

    def _compute_lambdas(self, prev_tokens):
        lambdas = []
        lambda_sum = 0
        for i in range(self._n - 1):
            c_prev = self.count(prev_tokens[i:])
            if c_prev == 0:
                lambdas.append(0)
            else:
                lambdas.append((1 - lambda_sum) * c_prev / (c_prev + self._gamma))
            lambda_sum += lambdas[-1]
        lambdas.append(1 - lambda_sum)
        return lambdas


class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """

        if beta is not None:
            # everything is training data
            train_sents = sents
            self._beta = beta
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        self._n = n
        self._addone = addone
        self._compute_counts(train_sents, all_ngrams=True)
        self._compute_A()

        if beta is None:
            # Grid search to find beta
            self._compute_beta(held_out_sents)

        self._compute_denoms()
        print('Selected beta:', self._beta)


    def _compute_A(self):
        self._A = a = defaultdict(set)
        for kgram in self._count.keys():
            if len(kgram) > 0 and kgram[-1] != START_TOKEN:
                a[kgram[:-1]].add(kgram[-1])

    def _compute_beta(self, sents):
        print('Finding optimal beta...')
        best_perplexity = math.inf
        best_beta = None
        for beta in np.linspace(BETA_MIN, BETA_MAX, BETA_LINSP_NUM):
            self._beta = beta
            self._compute_denoms()

            perplexity = self.perplexity(sents)
            if best_beta == None or best_perplexity > perplexity:
                best_beta = self._beta
                best_perplexity = perplexity
        self._beta = best_beta

    def _compute_denoms(self):
        self._denoms = defaultdict(float)
        # Normalization factors for every k-gram 0 < k < n
        for kgram in self._count.keys():
            if len(kgram) == 0 or len(kgram) == self._n:
                continue

            probs = [self.cond_prob(token, kgram[1:])
                     for token in self.A(kgram)]
            self._denoms[kgram] = 1 - sum(probs)

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self._A[tokens]

    def alpha(self, kgram):
        """Missing probability mass for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        len_A = len(self.A(kgram))
        if len_A:
            return self._beta * len_A / self.count(kgram)

        return 1

    def count(self, tokens):
        return self._count[tokens]

    def denom(self, kgram):
        return self._denoms.get(kgram, 1)

    def cond_prob(self, token, prev_tokens=None):
        if prev_tokens == () or prev_tokens is None:
            if self._addone:
                return (self.count((token,)) + 1) / (self.count(()) + self._V)
            else:
                return self.count((token,)) / self.count(())

        if token in self.A(prev_tokens):
            p = (self.count(prev_tokens + (token,)) - self._beta) / \
                self.count(prev_tokens)
        else:
            denom = self.denom(prev_tokens)
            p = self.alpha(prev_tokens) * \
                self.cond_prob(token, prev_tokens[1:]) / denom

        return p


class SentSorter():

    def __init__(self, model):
        self.model = model

    def get_most_probable_token(self, possible_tokens, prev_tokens):
        best_p = 0
        best_t = ''
        for token in possible_tokens:
            p = self.model.cond_prob(token, prev_tokens)
            if p > best_p:
                best_p = p
                best_t = token

        return best_t

    def sort_probable_sents(self, shuffled_sents):
        """ Get the most probable sentences out of the shuffled ones
            according to the model probabilities. """

        n = self.model._n
        sorted_sents = []
        for s in shuffled_sents:
            sent = s.copy()

            new_sent = ['<s>'] * (n - 1)
            while sent:
                next_tok = self.get_most_probable_token(sent, tuple(new_sent[-n + 1:]))
                new_sent += [next_tok]
                sent.remove(next_tok)
            new_sent = new_sent[n - 1:]
            sorted_sents.append(new_sent)

        return sorted_sents
