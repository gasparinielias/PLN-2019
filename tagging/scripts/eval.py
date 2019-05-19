"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:
  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from docopt import docopt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.consts import ANCORA_CORPUS_PATH


def print_results(model, tagged_sents, show_confusion_matrix):
    pred, ground_truth = [], []
    unknown = []
    for tsent in tagged_sents:
        sent, gt = zip(*tsent)
        pred += model.tag(sent)
        ground_truth += gt
        unknown += map(model.unknown, sent)

    pred = np.array(pred)
    ground_truth = np.array(ground_truth)
    unknown = np.array(unknown)
    known = np.invert(unknown)

    cm = confusion_matrix(ground_truth, pred)
    accuracy = cm.diagonal().sum() / cm.sum() * 100
    print("Accuracy: {:2.2f}%".format(accuracy))

    known_acc = sum(pred[known] == ground_truth[known]) / sum(known) * 100
    print("Accuracy for known words: {:2.2f}%".format(known_acc))

    unknown_acc = sum(pred[unknown] == ground_truth[unknown]) / sum(unknown) * 100
    print("Accuracy for unknown words: {:2.2f}%".format(unknown_acc))


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
    print_results(model, sents, opts['-c'])
