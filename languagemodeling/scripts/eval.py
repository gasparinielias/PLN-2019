"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from nltk.corpus import gutenberg


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    # WORK HERE!! LOAD YOUR EVALUATION CORPUS
    # sents = gutenberg.sents('austen-persuasion.txt')
    with open('sents', 'rb') as fp:
        [train_sents, test_sents] = pickle.load(fp)

    # compute the cross entropy
    log_prob = model.log_prob(test_sents)
    e = model.cross_entropy(test_sents)
    p = model.perplexity(test_sents)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
