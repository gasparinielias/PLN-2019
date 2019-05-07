"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from collections import defaultdict
from docopt import docopt
from itertools import chain

from tagging.ancora import SimpleAncoraCorpusReader


class POSStats():
    """Several statistics for a POS tagged corpus.
    """

    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        self.sents = list(tagged_sents)
        self._sent_count = len(self.sents)
        self._token_count = sum(len(sent) for sent in self.sents)

        word_tag_tuples = list(chain(*(sent for sent in self.sents)))
        word_types, tags = (set(x) for x in zip(*word_tag_tuples))

        self._word_types = word_types
        self._tags = tags

        tag_word_dict = defaultdict(set)
        self._tcount = tcount = defaultdict(lambda: defaultdict(int))
        self._word_freq = word_freq = defaultdict(int)
        self._tag_freq = tag_freq = defaultdict(int)
        for word, tag in word_tag_tuples:
            tag_word_dict[word].add(tag)
            tcount[tag][word] += 1
            word_freq[word] += 1
            tag_freq[tag] += 1

        self._word_ambiguity = word_ambiguity = defaultdict(list)
        for word in self._word_types:
            word_ambiguity[len(tag_word_dict[word])].append(word)

    def sent_count(self):
        """Total number of sentences."""
        return self._sent_count

    def token_count(self):
        """Total number of tokens."""
        return self._token_count

    def words(self):
        """Vocabulary (set of word types)."""
        return self._word_types

    def word_count(self):
        """Vocabulary size."""
        return len(self._word_types)

    def word_freq(self, w):
        """Frequency of word w."""
        return self._word_freq[w]

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""
        return self._word_ambiguity.get(1, ())

    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.

        n -- number of tags.
        """
        return self._word_ambiguity.get(n, ())

    def tags(self):
        """POS Tagset."""
        return self._tags

    def tag_count(self):
        """POS tagset size."""
        return len(self._tags)

    def tag_freq(self, t):
        """Frequency of tag t."""
        return self._tag_freq[t]

    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self._tcount[t])


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('corpus/ancora-3.0.1es/')
    sents = corpus.tagged_sents()

    # compute the statistics
    stats = POSStats(sents)

    print('Basic Statistics')
    print('================')
    print('sents: {}'.format(stats.sent_count()))
    token_count = stats.token_count()
    print('tokens: {}'.format(token_count))
    word_count = stats.word_count()
    print('words: {}'.format(word_count))
    print('tags: {}'.format(stats.tag_count()))
    print('')

    print('Most Frequent POS Tags')
    print('======================')
    tags = [(t, stats.tag_freq(t)) for t in stats.tags()]
    sorted_tags = sorted(tags, key=lambda t_f: -t_f[1])
    print('tag\tfreq\t%\ttop')
    for t, f in sorted_tags[:10]:
        words = stats.tag_word_dict(t).items()
        sorted_words = sorted(words, key=lambda w_f: -w_f[1])
        top = [w for w, _ in sorted_words[:5]]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(t, f, f * 100 / token_count, ', '.join(top)))
    print('')

    print('Word Ambiguity Levels')
    print('=====================')
    print('n\twords\t%\ttop')
    for n in range(1, 10):
        words = list(stats.ambiguous_words(n))
        m = len(words)

        # most frequent words:
        sorted_words = sorted(words, key=lambda w: -stats.word_freq(w))
        top = sorted_words[:5]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(n, m, m * 100 / word_count, ', '.join(top)))
