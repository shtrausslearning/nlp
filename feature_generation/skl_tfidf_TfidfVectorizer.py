''' TF-IDF features (TfidfVectorizer) '''
# corpus -> 3 lines -> 3 vectors
# length of vectors equal to the vocabulary length

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'Girl likes cat Tom',
    'Who likes the cat?',
    'Tom is a quiet cat'
]

vectoriser = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                             use_idf=True, smooth_idf=True)
vectors = vectoriser.fit_transform(corpus)
df_matrix = pd.DataFrame(vectors.toarray(),
                         columns=vectoriser.vocabulary_)
df_matrix.values

# array([[0.37311881, 0.63174505, 0.        , 0.4804584 , 0.        ,
#         0.        , 0.4804584 , 0.        ],
#        [0.34520502, 0.        , 0.        , 0.44451431, 0.        ,
#         0.5844829 , 0.        , 0.5844829 ],
#        [0.34520502, 0.        , 0.5844829 , 0.        , 0.5844829 ,
#         0.        , 0.44451431, 0.        ]])

vectoriser.vocabulary_

# {'girl': 1,
#  'likes': 3,
#  'cat': 0,
#  'tom': 6,
#  'who': 7,
#  'the': 5,
#  'is': 2,
#  'quiet': 4}
