''' Bag of Words Features (n-grams)'''
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

# ngram_range = (1,2) -> both unigrams & bigrams
vectoriser = CountVectorizer(ngram_range=(1,2))
vectors = vectoriser.fit_transform(corpus)

df_matrix = pd.DataFrame(vectors.toarray(),
                         columns=vectoriser.vocabulary_)
df_matrix.values

# array([[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
#        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

# vocabulary IDs
vectoriser.vocabulary_

# {'girl': 2,
#  'likes': 6,
#  'cat': 0,
#  'tom': 13,
#  'girl likes': 3,
#  'likes cat': 7,
#  'cat tom': 1,
#  'who': 15,
#  'the': 11,
#  'who likes': 16,
#  'likes the': 8,
#  'the cat': 12,
#  'is': 4,
#  'quiet': 9,
#  'tom is': 14,
#  'is quiet': 5,
#  'quiet cat': 10}
