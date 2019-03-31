from unittest import TestCase

from languagemodeling.ngram import SentSorter, AddOneNGram


class TestSentSorter(TestCase):

    def setUp(self):
        self.sents = [
            'el gato come'.split(),
            'la gata salta'.split()
        ]

    def test_get_most_probable_token_2gram(self):
        model = AddOneNGram(2, self.sents)
        ss = SentSorter(model)
        prev_tokens = [
            ('<s>',),
            ('el',),
            ('la',),
            ('gato',),
            ('gata',)
        ]

        possible_tokens = [
            ['gato', 'come', 'el'],
            ['gata', 'gato'],
            ['gato', 'salta', 'gata'],
            ['salta', 'gata', 'come'],
            ['la', 'el', 'salta']
        ]

        for p_tokens, prev in zip(possible_tokens, prev_tokens):
            self.assertEqual(
                ss.get_most_probable_token(p_tokens, prev_tokens),
                tok[-1])

    def test_sort_probable_sents_one_trainig_sent(self):
        sents = self.sents[:1]
        for i in range(2, 4):
            model = AddOneNGram(i, sents)
            unordered_sents = map(str.split, [
                'el come gato',
                'el gato come',
                'gato come el',
                'gato el come',
                'come el gato',
                'come gato el'
            ])
            ss = SentSorter(model)
            sorted_sents = ss.sort_probable_sents(unordered_sents)
            for sent in sorted_sents:
                self.assertEqual(sent, sents[0])
