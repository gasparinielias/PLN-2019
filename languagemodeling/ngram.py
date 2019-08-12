# https://docs.python.org/3/library/collections.html
from collections import defaultdict, Counter
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
        print('Computing counts...')
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
        self._compute_alpha()

    def _compute_A(self):
        self._A = a = defaultdict(set)
        for kgram in self._count.keys():
            if len(kgram) > 0 and kgram[-1] != START_TOKEN:
                a[kgram[:-1]].add(kgram[-1])

    def _compute_alpha(self):
        self._alpha = {}
        for kgram, A in self._A.items():
            len_A = len(A)
            if len_A:
                self._alpha[kgram] = self._beta * len_A / self.count(kgram)

    def _compute_beta(self, sents):
        print('Finding optimal beta...')
        best_perplexity = math.inf
        best_beta = None
        for beta in np.linspace(BETA_MIN, BETA_MAX, BETA_LINSP_NUM):
            self._beta = beta
            self._compute_denoms()
            self._compute_alpha()

            perplexity = self.perplexity(sents)
            if best_beta is None or best_perplexity > perplexity:
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
        return self._alpha.get(kgram, 1)

    def denom(self, kgram):
        return self._denoms.get(kgram, 1)

    def cond_prob(self, token, prev_tokens=None):
        if prev_tokens == () or prev_tokens is None:
            if self._addone:
                return (self.count((token,)) + 1) / (self.count(()) + self._V)
            else:
                return self.count((token,)) / self.count(())

        if self.count(prev_tokens + (token,)):
            p = (self.count(prev_tokens + (token,)) - self._beta) / \
                self.count(prev_tokens)
        else:
            alpha = self.alpha(prev_tokens)
            if not alpha:
                p = 0
            else:
                denom = self.denom(prev_tokens)
                p = self.alpha(prev_tokens) * \
                    self.cond_prob(token, prev_tokens[1:]) / denom

        return p


class SentSorter():

    def __init__(self, model):
        self.model = model

    def sort(self, shuffled_sent):
        """ table:
            (n-1)-gram: (p, counter, path)

            for token in sent:
                for ngram in prev_col:
                    if not available:
                        continue
                    p = p + model.cond_prob(token, ngram)
                    new_ngram = (ngram + (token,))[1:]
                    if new_ngram not in new_col or p < p:
                        new_col[new_ngram] = (new_p, available - 1, path + token)
                    
        """
        n = self.model._n
        self._pi = {
            0: {
                (START_TOKEN,) * (n - 1): (math.log2(1.0), Counter(shuffled_sent), ())
            }
        }

        for i in range(len(shuffled_sent)):
            self._pi[i + 1] = col = {}
            prev_col = self._pi[i]

            for ngram, (prev_p, counts, path) in prev_col.items():
                for token in counts:
                    if not counts[token]:
                        continue

                    p = prev_p + math.log2(self.model.cond_prob(token, ngram))
                    new_ngram = (ngram + (token,))[1:]
                    if new_ngram not in col or col[new_ngram][0] < p:
                        new_counts = counts.copy()
                        new_counts[token] -= 1
                        col[new_ngram] = (p, new_counts, path + (token,))

        max_path = None
        max_p = -math.inf
        last_col = self._pi[len(shuffled_sent)]
        for ngram, (prev_p, counts, path) in last_col.items():
            p = prev_p + math.log2(self.model.cond_prob(END_TOKEN, ngram))
            if max_path is None or p > max_p:
                max_p = p
                max_path = path

        return max_path
