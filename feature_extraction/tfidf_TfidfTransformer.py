from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def get_TFIDF(corpus):

    # BOW 
    cv = CountVectorizer(min_df=0.0, 
                         max_df=1.0)
    
    cv_matrix = cv.fit_transform(norm_corpus)

    # TF-IDF
    tt = TfidfTransformer(norm='l2', 
                          smooth_idf=True,
                          use_idf=True)

    tt_matrix = tt.fit_transform(cv_matrix)

    tt_matrix = tt_matrix.toarray()
    vocab = cv.get_feature_names_out()
    ldf = pd.DataFrame(tt_matrix, columns=vocab)
    return ldf.sort_index(axis=1)

# For comparison
get_TFIDF(corpus)

# array([[0.34520502, 0.5844829 , 0.        , 0.        , 0.44451431,
#         0.        , 0.5844829 ],
#        [0.50854232, 0.        , 0.        , 0.861037  , 0.        ,
#         0.        , 0.        ],
#        [0.34520502, 0.        , 0.5844829 , 0.        , 0.44451431,
#         0.5844829 , 0.        ]])
