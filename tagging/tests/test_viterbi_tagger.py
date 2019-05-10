# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from tagging.consts import START_TOKEN
from tagging.hmm import HMM, ViterbiTagger


class TestViterbiInit(TestCase):

    def setUp(self):
        self.n = 2
        self.tags = {'a', 'b', 'c'}
        self.all_ngrams = {
            ('a', 'a'),
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'a'),
            ('b', 'b'),
            ('b', 'c'),
            ('c', 'a'),
            ('c', 'b'),
            ('c', 'c')
        }

        trans = {
            (START_TOKEN,): {
                'a': 0.5,
                'b': 0.3,
                'c': 0.2
            },
            ('a',): {
                'a': 0.5,
                'b': 0.3,
                'c': 0.2
            },
            ('b',): {
                'a': 0.5,
                'b': 0.3,
                'c': 0.2
            },
            ('c',): {
                'a': 0.5,
                'b': 0.3,
                'c': 0.2
            }
        }

        out = {
            'a': {
                'w1': 0.3,
                'w2': 0.7
            },
            'b': {
                'w1': 0.8,
                'w2': 0.2
            },
            'c': {
                'w1': 0.3,
                'w2': 0.7
            }
        }
        self.hmm1 = HMM(1, self.tags, trans, out)
        self.hmm2 = HMM(2, self.tags, trans, out)

    def test_all_tag_ngrams(self):
        viterbi = ViterbiTagger(self.hmm2)
        viterbi_ngrams = viterbi.all_tag_ngrams(self.n)

        self.assertEqual(self.all_ngrams, set(viterbi_ngrams))

    def test_init_pi_long_prefix(self):
        viterbi = ViterbiTagger(self.hmm2)
        prefix = ['w1'] * self.n
        viterbi.init_pi(prefix)

        for tag_ngram in self.tags:
            ngram_p = self.hmm2.tag_log_prob(tag_ngram, add_end_token=False) + \
                      log2(self.hmm2.out_prob('w1', tag_ngram[-1]))
            self.assertAlmostEqual(ngram_p, viterbi.pi[0][(tag_ngram,)])

    def test_init_pi_short_prefix(self):
        viterbi = ViterbiTagger(self.hmm1)
        prefix = []
        viterbi.init_pi(prefix)

        self.assertEqual(len(viterbi.pi[0].keys()), 1)
        self.assertAlmostEqual(viterbi.pi[0][()], log2(1))

    def test_fill_column(self):
        viterbi = ViterbiTagger(self.hmm2)
        prefix = ['w1'] * self.n
        viterbi.init_pi(prefix)
        viterbi.fill_column(0, 'w2')
        # TODO backpointers
        self.assertTrue(False)
        

class TestViterbiTagger(TestCase):

    def setUp(self):
        self.tagset = {'D', 'N', 'V'}
        self.out = {
            'D': {'the': 1.0},
            'N': {'dog': 0.4, 'barks': 0.6},
            'V': {'dog': 0.1, 'barks': 0.9},
        }

    def test_tag(self):
        tagset = self.tagset
        out = self.out

        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 1.0},
            ('D', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.4), ['D', 'N']),
            },
            3: {
                ('N', 'V'): (log2(0.4 * 0.9), ['D', 'N', 'V']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def test_tag2(self):
        tagset = self.tagset
        out = self.out

        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 1.0},
            ('D', 'N'): {'V': 0.8, 'N': 0.2},
            ('N', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.4), ['D', 'N']),
            },
            3: {
                ('N', 'V'): (log2(0.8 * 0.4 * 0.9), ['D', 'N', 'V']),
                ('N', 'N'): (log2(0.2 * 0.4 * 0.6), ['D', 'N', 'N']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def test_tag3(self):
        tagset = self.tagset
        out = self.out

        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 0.8, 'V': 0.2},
            ('D', 'N'): {'V': 0.8, 'N': 0.2},
            ('D', 'V'): {'V': 0.8, 'N': 0.2},
            ('N', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
            # XXX: not sure if needed
            ('V', 'N'): {'</s>': 1.0},
            ('V', 'V'): {'</s>': 1.0},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.8 * 0.4), ['D', 'N']),
                ('D', 'V'): (log2(0.2 * 0.1), ['D', 'V']),
            },
            3: {
                ('N', 'V'): (log2(0.8 * 0.4 * 0.8 * 0.9), ['D', 'N', 'V']),
                ('N', 'N'): (log2(0.8 * 0.4 * 0.2 * 0.6), ['D', 'N', 'N']),
                ('V', 'V'): (log2(0.2 * 0.1 * 0.8 * 0.9), ['D', 'V', 'V']),
                ('V', 'N'): (log2(0.2 * 0.1 * 0.2 * 0.6), ['D', 'V', 'N']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1, tags1 = d1[k2]
                prob2, tags2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
                self.assertEqual(tags1, tags2)
