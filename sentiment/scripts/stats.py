from os.path import join

from sentiment.consts import INTERTASS_DIR
from sentiment.tass import InterTASSReader

def get_corpus_stats(corpus_name):
    print('Stats for {}:'.format(corpus_name))

    train_corpus = join(
        join(INTERTASS_DIR, corpus_name),
        '-'.join(['intertass', corpus_name, 'train', 'tagged']) + '.xml')

    reader = InterTASSReader(train_corpus)
    amnt_tweets = 0
    tags = {
        'P': 0,
        'N': 0,
        'NEU': 0,
        'NONE': 0
    }

    for tweet in reader.tweets():
        amnt_tweets += 1
        tag = tweet['sentiment']
        tags[tag] += 1

    print('\tTweets for training: {}'.format(amnt_tweets))
    for tag, value in tags.items():
        print('\tTweets with sentiment {}: {}'.format(tag, value))

if __name__ == '__main__':
    get_corpus_stats('PE')
    get_corpus_stats('CR')
    get_corpus_stats('ES')
