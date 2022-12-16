''' Ordinal Encoder '''

from sklearn.preprocessing import OrdinalEncoder

words = np.array([['NLP','is','awesome','eh','NLP'],
                  ['NLP','is','very','interesting','eh']]).T

encoder = OrdinalEncoder()
corpus_encoded = encoder.fit_transform(words)
corpus_encoded

# array([[0., 0.],
#        [3., 3.],
#        [1., 4.],
#        [2., 2.],
#        [0., 1.]])

encoder.categories_

# [array(['NLP', 'awesome', 'eh', 'is'], dtype='<U11'),
#  array(['NLP', 'eh', 'interesting', 'is', 'very'], dtype='<U11')]
