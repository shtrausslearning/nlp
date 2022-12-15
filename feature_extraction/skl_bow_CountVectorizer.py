''' Bag of Words Features'''
# Vertical direction -> number of sentences in corpus
# Horizontal diction -> Vocabulary ID 
# Value represents the number of words found in the sentence

from sklearn.feature_extraction.text import CountVectorizer

# corpus with multiple entries
corpus = [
    'Girl likes cat Tom',
    'Who likes the cat?',
    'Tom is a quiet cat'
]

vectoriser = CountVectorizer()
vectors = vectoriser.fit_transform(corpus)

df_matrix = pd.DataFrame(vectors.toarray(),
                         columns=vectoriser.vocabulary_)
df_matrix.values

# array([[1, 1, 0, 1, 0, 0, 1, 0],
#        [1, 0, 0, 1, 0, 1, 0, 1],
#        [1, 0, 1, 0, 1, 0, 1, 0]])

df_matrix.vocabulary_

# {'girl': 1,
#  'likes': 3,
#  'cat': 0,
#  'tom': 6,
#  'who': 7,
#  'the': 5,
#  'is': 2,
#  'quiet': 4}
