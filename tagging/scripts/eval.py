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


def plot_confusion_matrix(cm, labels):
    cmap=plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title='Top-frequent-tags confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.png')

def print_results(model, tagged_sents, show_confusion_matrix):
    pred, ground_truth = [], []
    unknown = []
    for tsent in tagged_sents:
        sent, gt = zip(*tsent)
        pred += model.tag(sent)
        ground_truth += gt
        unknown += map(model.unknown, sent)

    labels, counts = np.unique(ground_truth + pred, return_counts=True)
    pred = np.array(pred)
    ground_truth = np.array(ground_truth)
    unknown = np.array(unknown)
    known = np.invert(unknown)

    cm = confusion_matrix(ground_truth, pred, labels)
    accuracy = cm.diagonal().sum() / cm.sum() * 100
    print("Accuracy: {:2.2f}%".format(accuracy))

    known_acc = (pred[known] == ground_truth[known]).sum() / known.sum() * 100
    print("Accuracy for known words: {:2.2f}%".format(known_acc))

    unknown_acc = (pred[unknown] == ground_truth[unknown]).sum() / unknown.sum() * 100
    print("Accuracy for unknown words: {:2.2f}%".format(unknown_acc))

    if show_confusion_matrix:
        top = 5
        top_tags = np.argsort(-counts)[:top]
        labels = labels[top_tags]

        cm = cm.astype('float') / cm.sum()
        cm = cm[top_tags][:, top_tags]

        plot_confusion_matrix(cm, labels)


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
