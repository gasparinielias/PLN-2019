import re

from sentiment.consts import URL_TOKEN, LAUGH_TOKEN

class preprocessor():
    def __call__(self, doc):
        tagged_users_pat = r'@\w+'
        urls_pat = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ' + \
                   ']|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        vocals_rep_pat = r'(\w)\1{1,}'
        laugh_pat = r'\b[ja]+\b'
        return re.sub(tagged_users_pat, '',
               re.sub(laugh_pat, LAUGH_TOKEN,
               re.sub(urls_pat, URL_TOKEN,
               re.sub(vocals_rep_pat, '\g<1>',
               doc.lower()))))
