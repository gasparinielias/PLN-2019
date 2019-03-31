"""Get the most probable order of a set of sentences

Usage:
    sent_sort.py -i <file>
    sent_sort.py -h | --help

Options:
    -i <file>   Input model file.
    -h --help   Show this screen.
"""

import nltk
import numpy as np
import pickle
from docopt import docopt

from languagemodeling.ngram import SentSorter

if __name__ == '__main__':
    opts = docopt(__doc__)
    with open(opts['-i'], 'rb') as fp:
        model = pickle.load(fp)

    with open('sents', 'rb') as fp:
        [train, test] = pickle.load(fp)

    for _ in map(np.random.shuffle, test): pass

    ss = SentSorter(model)
    sorted_sents = ss.sort_probable_sents(test)

    avg_edit_distance = [nltk.edit_distance(x, y) for x, y in zip(sorted_sents, test)]
    avg_edit_distance = sum(avg_edit_distance) / len(avg_edit_distance)
    print("Average edit distance: {}".format(avg_edit_distance))
