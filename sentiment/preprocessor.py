import re
from sentiment.consts import URL_TOKEN, LAUGH_TOKEN


class Preprocessor():
    def __call__(self, doc):
        tagged_users_pat = r'@\w+'
        laugh_pat = r'\b[ja]*j[ja]*\b'
        vocals_rep_pat = r'(\w)\1{1,}'
        urls_pat = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ' + \
                   r']|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        return re.sub(tagged_users_pat, '',
               re.sub(laugh_pat, LAUGH_TOKEN,
               re.sub(vocals_rep_pat, '\g<1>',
               re.sub(urls_pat, URL_TOKEN,
               doc.lower()))))
