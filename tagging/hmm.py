import math
from collections import defaultdict

from tagging.consts import START_TAG, END_TAG


class HMM():

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        assert n > 0

        self._n = n
        self._tagset = tagset
        self._trans = trans
        self._out = out
        self._tagger = None

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

    def trans_log_prob(self, tag, prev_tags=None):
        t_prob = self.trans_prob(tag, prev_tags)
        if t_prob > 0:
            return math.log2(t_prob)

        return -math.inf

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self._out.get(tag, {}).get(word, 0)

    def out_log_prob(self, word, tag):
        out_prob = self.out_prob(word, tag)
        if out_prob > 0:
            return math.log2(out_prob)

        return -math.inf

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        p = 1
        n = self._n
        y = (START_TAG,) * (n - 1) + tuple(y) + (END_TAG,)
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
        y = (START_TAG,) * (n - 1) + tuple(y) + (END_TAG,) * add_end_token
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
        if not self._tagger:
            self._tagger = ViterbiTagger(self)

        return self._tagger.tag(sent)


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self._n = n
        self._addone = addone

        self._tagset, self._vocab = self.compute_tagset_vocabulary(tagged_sents)

        self._out_counts, self._tcounts = self.compute_counts(tagged_sents)

    def compute_counts(self, tagged_sents):
        n = self._n
        ngram_counts = defaultdict(int)
        out_counts = defaultdict(lambda: defaultdict(int))
        for tsent in tagged_sents:
            for word, tag in tsent:
                out_counts[tag][word] += 1

            _, tags = zip(*tsent)
            tags = (START_TAG,) * (n - 1) + tuple(tags) + (END_TAG,)

            for i in range(len(tags) - n + 1):
                ngram = tags[i:i + n]
                ngram_counts[ngram] += 1
                ngram_counts[ngram[:-1]] += 1

        return dict(out_counts), dict(ngram_counts)

    def compute_tagset_vocabulary(self, tagged_sents):
        tagset = set()
        vocab = set()
        for tsent in tagged_sents:
            sent, tags = zip(*tsent)
            tagset.update(tags)
            vocab.update(sent)
        return tagset, vocab

    def trans_prob(self, tag, prev_tags=None):
        assert prev_tags is not None or self._n > 0

        prev_tags = prev_tags or ()
        addone = self._addone

        p = (self.tcount(prev_tags + (tag,)) + addone) / \
            (self.tcount(prev_tags) + self.tagset_size() * addone)

        return p

    def out_prob(self, word, tag):
        if self.unknown(word):
            return 1 / self.vocab_size()

        return self.wcount(word, tag) / self.tag_count(tag)

    def tcount(self, tokens):
        """Count for an n-gram or (n-1)-gram of tags.

        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        return self._tcounts.get(tokens, 0)

    def tag_count(self, tag):
        """Times of occurence of "tag"
        """
        return sum(self._out_counts.get(tag, {}).values())

    def wcount(self, word, tag):
        return self._out_counts.get(tag, {}).get(word, 0)

    def vocab_size(self):
        return len(self._vocab)

    def tagset_size(self):
        return len(self._tagset)

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._vocab


class ViterbiTagger():

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self._model = hmm
        self.tagset = hmm.tagset()
        self._pi = None

    def tag(self, sent):
        self.init_pi()
        for i in range(0, len(sent)):
            self.fill_column(i + 1, sent[i])

        path = None
        max_p = -math.inf
        last_col = self._pi[len(sent)]
        for prev_tags, (prob, tokens) in last_col.items():
            new_p = prob + self._model.trans_log_prob(END_TAG, prev_tags)
            if new_p > max_p:
                max_p = new_p
                path = tokens

        return path

    def init_pi(self):
        n = self._model._n
        self._pi = {
            0: {
                (START_TAG,) * (n - 1): (math.log2(1.0), [])
            }
        }

    def fill_column(self, i, token):
        tagset = self.tagset
        self._pi[i] = col = {}
        prev_col = self._pi[i - 1]
        for prev_tags in prev_col:
            for tag in tagset:
                p = prev_col[prev_tags][0] + \
                    self._model.trans_log_prob(tag, prev_tags) + \
                    self._model.out_log_prob(token, tag)

                if p != -math.inf:
                    new_tags = prev_tags[1:] + (tag,)
                    col[new_tags] = (p, prev_col[prev_tags][1] + [tag])
