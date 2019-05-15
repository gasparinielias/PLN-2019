"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:
  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.consts import ANCORA_CORPUS_PATH


def print_results(model, sents):
    pred_true_tags = defaultdict(lambda: defaultdict(int))
    true_tags = defaultdict(int)
    unk_words = 0
    total_unk_words = 0
    known_words = 0
    total_known_words = 0

    for sent in sents:
        pred = model.tag(sent)
        for i, pred_tag in enumerate(pred):
            word, tag = sent[i]
            pred_true_tags[pred_tag][tag] += 1
            true_tags[tag] += 1
            if model.unknown(word):
                total_unk_words += 1
                if pred_tag == tag:
                    unk_words += 1
            else:
                total_known_words += 1
                if pred_tag == tag:
                    known_words += 1

    tp = 0
    for tag in true_tags:
        tp += pred_true_tags[tag][tag]

    acc = tp / sum(true_tags.values()) * 100

    print("Accuracy: {:2.2f}%".format(acc))
    if total_known_words:
        print("Accuracy for known words: {:2.2f}%".format(
            known_words / total_known_words * 100))

    if total_unk_words:
        print("Accuracy for unknown words: {:2.2f}%".format(
            unk_words / total_unk_words * 100))


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(ANCORA_CORPUS_PATH, files)
    sents = list(corpus.tagged_sents())

    # tag and evaluate
    print_results(model, sents)
