"""Train a sequence tagger.

Usage:
  train.py [options] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
                  twc: Three words classifier
  -o <file>     Output model file.
  -h --help     Show this screen.
  -n <int>      n-gram (MLHMM only)
  -c <clf>      Classifier to use
"""
from docopt import docopt
import pickle

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger
from tagging.classifier import ClassifierTagger
from tagging.consts import ANCORA_CORPUS_PATH
from tagging.hmm import MLHMM


models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    'mlhmm': MLHMM,
    'twc': ClassifierTagger
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(ANCORA_CORPUS_PATH, files)
    sents = corpus.tagged_sents()

    # train the model
    model_class = models[opts['-m']]
    if opts['-m'] == 'mlhmm':
        model = model_class(int(opts['-n']), sents)
    else:
        model = model_class(sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
