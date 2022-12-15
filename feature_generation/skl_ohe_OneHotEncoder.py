''' One-Hot Encoding Features'''
# Vertical direction -> number of words in the list (sentences are treated as words) 
# Horizontal direction -> unique name in list 
# Value represents the location where the word occurs 
# (eg. awesome -> 3rd & last string, 1 value @ its identifier location -> 3rd column (3rd in categories_))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

words = np.array(['NLP','is','awesome','eh','NLP today','awesome'])

encoder = OneHotEncoder(sparse=False)
vectors = encoder.fit_transform(words[:,None])
vectors

df_matrix = pd.DataFrame(vectors,columns=encoder.categories_)
df_matrix

# array([[1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1.],
#        [0., 0., 1., 0., 0.],
#        [0., 0., 0., 1., 0.],
#        [0., 1., 0., 0., 0.],
#        [0., 0., 1., 0., 0.]])
