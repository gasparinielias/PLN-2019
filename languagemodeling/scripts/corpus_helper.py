import os

from nltk.tokenize import word_tokenize, sent_tokenize


CORPUS_DIR = 'corpus'

def load_corpus(fname):
    with open(os.path.join(CORPUS_DIR, fname), 'r') as fp:
        data = fp.read()
        sents = [[w.lower() for w in word_tokenize(sent)] for sent in sent_tokenize(data)]
    return sents
