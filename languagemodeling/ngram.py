# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math

from languagemodeling.consts import START_TOKEN, END_TOKEN


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
        wc = sum(map(len, sents)) + len(sents) # each sent lacking of END_TOKEN
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

        count = defaultdict(int)

        # WORK HERE!!
        for sent in sents:
            nsent = [START_TOKEN] * (n - 1) + sent + [END_TOKEN]
            for i in range(len(nsent) - n + 1):
                ngram = tuple(nsent[i:i + n])
                nm1gram = tuple(nsent[i:i + n - 1])
                count[ngram] += 1
                count[nm1gram] += 1
        self._count = dict(count)

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
        assert((prev_tokens is not None) or self._n == 1)

        if prev_tokens is None:
            prev_tokens = ()

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
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        for sent in sents:
            for word in sent:
                voc.add(word)
        voc.add(END_TOKEN)

        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        assert((prev_tokens is not None) or self._n == 1)

        if prev_tokens is None:
            prev_tokens = ()

        return (self.count(prev_tokens + (token,)) + 1) / (self.count(prev_tokens) + self._V)


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
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        self._count = defaultdict(int)
        for sent in train_sents:
            self._count[()] += len(sent) + 1
            sent = [START_TOKEN] * (n - 1) + sent + [END_TOKEN]

            # Count all i-grams
            for i in range(1, n + 1):
                for j in range(n - i, len(sent) - i + 1):
                    self._count[tuple(sent[j:j + i])] += 1

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            for sent in train_sents:
                for word in sent:
                    voc.add(word)
            voc.add(END_TOKEN)

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma

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
        assert((prev_tokens is not None) or self._n == 1)

        if prev_tokens is None:
            prev_tokens = ()

        # Compute lambdas
        lambdas = []
        for i in range(self._n - 1):
            c_prev = self.count(prev_tokens[i:])
            if c_prev + self._gamma == 0:
                lambdas.append(0)
            else:
                lambdas.append((1 - sum(lambdas)) * c_prev / \
                    (c_prev + self._gamma))
        lambdas.append(1 - sum(lambdas))

        p = 0
        for i in range(self._n):
            if lambdas[i] != 0:
                p += lambdas[i] * self.count(prev_tokens[i:] + (token,)) / \
                    self.count(prev_tokens[i:])
        return p
