grid_params = {
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
            'clf__C': [.1, .2, .3, .4, .5, .6],
            'clf__class_weight': ['balanced'],
        }
    ],
    'mnb': [
        {
            'clf__alpha': [.5, .6, .7, 1],
        }
    ]
}
