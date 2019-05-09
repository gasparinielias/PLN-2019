import math

from tagging.consts import START_TOKEN, END_TOKEN


class HMM():

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self._n = n
        self._tagset = tagset
        self._trans = trans
        self._out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self._tagset

    def trans_prob(self, tag, prev_tags=None):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        assert prev_tags is not None or self._n == 1
        prev_tags = prev_tags or ()

        return self._trans.get(prev_tags, {}).get(tag, 0)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self._out.get(tag, {}).get(word, 0)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        p = 1
        n = self._n
        y = (START_TOKEN,) * (n - 1) + tuple(y) + (END_TOKEN,)
        for i in range(len(y) - self._n + 1):
            tag = y[i + n - 1]
            prev_tags = y[i:i + n - 1]
            p *= self.trans_prob(tag, prev_tags)

        return p

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        p = self.tag_prob(y)
        for i in range(len(y)):
            p *= self.out_prob(x[i], y[i])

        return p

    def tag_log_prob(self, y, add_end_token=True):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        p = 0
        n = self._n
        y = (START_TOKEN,) * (n - 1) + tuple(y) + (END_TOKEN,) * add_end_token
        for i in range(len(y) - self._n + 1):
            tag = y[i + n - 1]
            prev_tags = y[i:i + n - 1]
            if self.trans_prob(tag, prev_tags) == 0:
                return -math.inf

            p += math.log2(self.trans_prob(tag, prev_tags))

        return p

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        p = self.tag_log_prob(y)
        for i in range(len(y)):
            if self.out_prob(x[i], y[i]) == 0:
                return -math.inf

            p += math.log2(self.out_prob(x[i], y[i]))

        return p

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        return []


class ViterbiTagger():

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self._model = hmm

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        n = self._model._n
        self.init_mat(sent[:n])

    def init_mat(self, sent_prefix):
        tag_ngrams = self.all_tag_ngrams(min(self._model._n - 1, len(sent_prefix)))

        first_col = {}
        for ngram in tag_ngrams:
            first_col[ngram] = self._model.tag_log_prob(ngram, add_end_token=False)
            for i, tag in enumerate(ngram):
                p = self._model.out_prob(sent_prefix[i], tag)
                if p != 0:
                    first_col[ngram] += math.log2(p)
                else:
                    first_col[ngram] = -math.inf
                    break
        self.mat = [first_col, {}]

    def all_tag_ngrams(self, max_length):
        if max_length == 0:
            return [()]

        res = []
        tags = list(self._model.tagset())
        for tag in tags:
            res.extend(prev + (tag,) for prev in
                       self.all_tag_ngrams(max_length - 1))

        return res
