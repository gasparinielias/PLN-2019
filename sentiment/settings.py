from sentiment.consts import POL_THRESH_LOW, POL_THRESH_HIGH

GRID_PARAMS = {
    'svm': [
        {
            'clf__C': [.1, .2, .3, .4, .5, .6],
        },
        {
            'clf__C': [.1, .2, .3, .4, .5, .6],
            'clf__dual': [False],
            'clf__penalty': ['l1'],
        }
    ],
    'maxent': [
        {
            'vect__polarities__count__thresh': [
                [x / 10, y / 10] for x in POL_THRESH_LOW
                                 for y in POL_THRESH_HIGH],
            'clf__C': [.1, 1, 10, 100, 1000, 5000],
            'clf__class_weight': ['balanced'],
            'clf__solver': ['lbfgs'],
            'clf__multi_class': ['auto'],
            'clf__max_iter': [1000]
        }
    ],
    'mnb': [
        {
            'clf__alpha': [.5, .6, .7, 1],
        }
    ]
}
