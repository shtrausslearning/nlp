'''

Generate & Prepare Corpus
simple normalisation function

''' 


import pandas as pd
import numpy as np
from collections import Counter
import nltk
import re
from numpy.linalg import norm

''' Create Data '''

# corpus (several documents)
corpus = ['Girl likes cat Tom',
          'I like the cat',
          'The cat likes to stay at home']

corpus = pd.DataFrame(corpus,columns=['text'])
corpus = corpus['text']


''' Normalise Text '''
# TfidfTransformer will remove stop words & tokenise the data

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalise(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalise)
norm_corpus = normalize_corpus(corpus)
# array(['girl likes cat tom', 'like cat', 'cat likes stay home'],

'''

Recreation of TfidfTransformer

- Compute the **<span style='color:#FFC300'>Term frequencies</span>** (TF) for the corpus (BOW)
- Compute the **<span style='color:#FFC300'>Document frequency</span>** (# of documents the term exists)
- Compute the **<span style='color:#FFC300'>Inverse Document Frequency</span>** & set it to the diagonal component
- Compute the **<span style='color:#FFC300'>TF-IDF</span>** matrix (using matrix multiplication)
- Compute the normalised **<span style='color:#FFC300'>TF-IDF</span>** matrix

'''

import re
from numpy.linalg import norm
import scipy.sparse as sp

def get_TFIDF2(corpus,verbose=1):

    ''' Define Vocabulary Dictionary '''
    # create a blank dictionary

    lst_words = list(set([word for doc in [doc.split() 
                                           for doc in corpus] 
                             for word in doc]))
    # initialise dict
    def_feature_dict = {w: 0 for w in lst_words}

    ''' Compute the Term Frequencies (TF) (BOW)'''
    # compute Bag of Words 

    BoW = []
    for doc in corpus:
        bow_feature_doc = Counter(doc.split())
        all_features = Counter(def_feature_dict)
        bow_feature_doc.update(all_features)
        BoW.append(bow_feature_doc)
        
    BoW = pd.DataFrame(BoW)
    np_BoW = np.array([BoW])
    if(verbose == 1):
        print(f'BoW:\n {np_BoW}','\n')

    ''' Compute the Document Frequencies (DF) '''
    # with smoothing (df=df+1)

    features = list(BoW.columns)
    df = np.diff(sp.csc_matrix(BoW, copy=True).indptr)
    df = 1 + df
    
    if(verbose == 1):
        print(f'DF:\n {df}','\n')

    ''' Compute IDF (inverse document frequencies) '''
    # with smoothing (1+ ...)
    
    total_docs = 1 + len(corpus)
    idf = 1.0 + np.log(float(total_docs) / df)
    
    if(verbose == 1):
        print(f'IDF:\n {idf}','\n')

    ''' Add computed IDF terms to diagonal '''
    # so we can use it in matrix mult TF,IDF

    total_features = BoW.shape[1]
    idf_diag = sp.spdiags(idf, diags=0, 
                          m=BoW.shape[1], 
                          n=BoW.shape[1])
    idf_dense = idf_diag.todense()

    ''' Compute TF-IDF feature matrix '''

    tf = np.array(BoW, dtype='float64')
    tfidf = np.matmul(tf,idf_dense)
    
    if(verbose == 1):
        print(f'TFIDF:\n {tfidf}','\n')
    
    ''' Compute Matrix L2 Norm '''
    # If required L2 Normalisation

    norms = norm(tfidf, axis=1)
    norm_tfidf = tfidf / norms[:, None]
    
    if(verbose == 1):
        print(f'L2 TFIDF:\n {norm_tfidf}','\n')

    ldf = pd.DataFrame(norm_tfidf,columns=features)
    ldf.sort_index(axis=1)
    print('Sorted Data')
    print(ldf.values)

get_TFIDF2(norm_corpus)

# BoW:
#  [[[1 1 1 1 0 0 0]
#   [0 0 1 0 1 0 0]
#   [0 1 1 0 0 1 1]]] 

# DF:
#  [2 3 4 2 2 2 2] 

# IDF:
#  [1.69314718 1.28768207 1.         1.69314718 1.69314718 1.69314718
#  1.69314718] 

# TFIDF:
#  [[1.69314718 1.28768207 1.         1.69314718 0.         0.
#   0.        ]
#  [0.         0.         1.         0.         1.69314718 0.
#   0.        ]
#  [0.         1.28768207 1.         0.         0.         1.69314718
#   1.69314718]] 

# L2 TFIDF:
#  [[0.5844829  0.44451431 0.34520502 0.5844829  0.         0.
#   0.        ]
#  [0.         0.         0.50854232 0.         0.861037   0.
#   0.        ]
#  [0.         0.44451431 0.34520502 0.         0.         0.5844829
#   0.5844829 ]] 

# [[0.5844829  0.44451431 0.34520502 0.5844829  0.         0.
#   0.        ]
#  [0.         0.         0.50854232 0.         0.861037   0.
#   0.        ]
#  [0.         0.44451431 0.34520502 0.         0.         0.5844829
#   0.5844829 ]]
