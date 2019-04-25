"""Train a Sentiment Analysis model.

Usage:
  train.py [options]
  train.py -h | --help

Options:
  -m <model>    Model to use [default: basemf]:
                  basemf: Most frequent sentiment
                  clf: Machine Learning Classifier
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  --all         Use develop as training [default: False]
  -h --help     Show this screen.
"""
import numpy as np
import pickle
from docopt import docopt

from sentiment.tass import InterTASSReader
from sentiment.baselines import MostFrequent
from sentiment.classifier import SentimentClassifier


models = {
    'basemf': MostFrequent,
    'clf': SentimentClassifier,
}

corpus_files = [
    'corpus/InterTASS/PE/intertass-PE-{}-tagged.xml',
    'corpus/InterTASS/CR/intertass-CR-{}-tagged.xml',
    'corpus/InterTASS/ES/intertass-ES-{}-tagged.xml',
]

models_output = [
    'models/PE-{}.model',
    'models/CR-{}.model',
    'models/ES-{}.model'
]


if __name__ == '__main__':
    opts = docopt(__doc__)

    for i in range(len(corpus_files)):
        # load corpora
        reader = InterTASSReader(corpus_files[i].format('train'))
        X, y = list(reader.X()), list(reader.y())
        reader = InterTASSReader(corpus_files[i].format('development'))
        X_dev, y_dev = list(reader.X()), list(reader.y())

        train = list(range(len(X)))
        test = list(range(len(X), len(X) + len(X_dev)))
        X_all = X + X_dev
        y_all = y + y_dev

        # train model
        model_type = opts['-m']
        if model_type == 'clf':
            model = models[model_type](clf=opts['-c'])
        else:
            model = models[model_type]()  # baseline

        model.fit(X_all, y_all, train, test,
                  grid_search=True, refit_all=opts['--all'])

        # save model
        f = open(models_output[i].format(opts['-c']), 'wb')
        pickle.dump(model, f)
        f.close()
