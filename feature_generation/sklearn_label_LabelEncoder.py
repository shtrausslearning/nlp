''' Label Encoding '''
from sklearn.preprocessing import LabelEncoder

paragraph = "write paragaraph here to convert into tokens."
lst_paragraph = paragraph.split(' ')

label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(lst_paragraph)
corpus_encoded

# array([6, 3, 1, 4, 0, 2, 5])

label_encoder.classes_

# array(['convert', 'here', 'into', 'paragaraph', 'to', 'tokens.', 'write'],
#       dtype='<U10')
