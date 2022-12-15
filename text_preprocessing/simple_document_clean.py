'''

# Input list of documents (corpus)
------------------------------------
# - spacy tokenisation
# - text normalisation (lowering)
# - lemmatisation of documents
# - filter stopwords & punctuations

# Output list of cleaned documents (corpus)
'''

# Input corpus list
corpus = ['Evidently someone with the authority to make decisions has arrived.',
          'I think I smell the stench of your cologne, Agent Cooper.',
          'Smells like hubris.']

import spacy
import pandas as pd
import string
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punctuations = string.punctuation

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, verbose=False):
    
    texts = []
    for ii,doc in enumerate(docs):
    
        if(ii % 1000 == 0 and verbose):
            print(f"Processed {ii+1} out of {len(docs)} documents.")
        
        # Load statistical model
        nlp = spacy.load("en_core_web_sm",
                         disable=['parser', 'ner'])
        doc = nlp(doc)
        
        # choose tokens which are not pronouns (pos_)
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ != 'PRON'] 

        # choose tokens which are not punctuations (token) 
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]

        tokens = ' '.join(tokens)
        texts.append(tokens)
        
    return texts

print(f'unprocessed: \n{corpus}')

processed = cleanup_text(corpus)
print(f'\nprocessed: \n{processed}')

# unprocessed: 
# ['Evidently someone with the authority to make decisions has arrived.', 'I think I smell the stench of your cologne, Agent Cooper.', 'Smells like hubris.']

# processed: 
# ['evidently authority make decision arrive', 'think smell stench cologne agent cooper', 'smell like hubris']
