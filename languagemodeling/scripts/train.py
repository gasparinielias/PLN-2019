"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file> [-a] [-g <gamma>]
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -a            Addone = True (only for InterpolatedNGram model)
  -g <gamma>    Gamma (InterpolatedNGram model) [default: None]
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import numpy as np

from nltk.corpus import gutenberg

from languagemodeling.consts import MODELS_DIR, SEED, TRAIN_PER
from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram
from languagemodeling.scripts import corpus_helper


models = {
    'ngram': NGram,
    'addone': AddOneNGram,
    'inter': InterpolatedNGram
}

def get_sents(load_sents):
    if load_sents:
        with open('sents', 'rb') as fp:
            train_sents, test_sents = pickle.load(fp)
    else:
        # load the data
        sents = corpus_helper.load_corpus('lavoz.corpus')
        sents = np.unique(sents)
        np.random.seed(SEED)
        np.random.shuffle(sents)

        M = len(sents)
        t = int(M * TRAIN_PER)
        train_sents = sents[:t]
        test_sents = sents[t:]
        with open('sents', 'wb') as fp:
            pickle.dump([train_sents, test_sents], fp)
    return train_sents, test_sents


if __name__ == '__main__':
    opts = docopt(__doc__)

    load_sents = False
    train_sents, test_sents = get_sents(load_sents)

    # train the model
    n = int(opts['-n'])
    model_class = models[opts['-m']]
    addone = '-a' in opts.keys()
    gamma = None if opts['-g'] == 'None' else float(opts['-g'])

    if opts['-m'] != 'inter':
        model = model_class(n, train_sents)
    else:
        model = model_class(n, train_sents, gamma=gamma, addone=addone)

    # save it
    filename = opts['-o']
    f = open(filename + '-%s.model' % n, 'wb')
    pickle.dump(model, f)
    f.close()
