
## SpaCy library

### 1 | Defining statistical model & input

Load statistical model [`.load(model)`](https://spacy.io/usage/models)

```python
# nlp = spacy.load("ru_core_news_sm")
nlp = spacy.load("ru_core_news_lg")
# nlp = spacy.load('en_core_web_sm')
```

Define string `nlp(string)`

```python
doc = nlp('Evidently someone with the authority to make decisions has arrived.')
```
### 2 | Pipeline Output

Upon defining `nlp()`, we have access to:

- `.text` : Tokens
- `.lemma_` : lemmatised tokens
- `.sents` Sentences
- `.pos_` : Coarse `POS` tags
- `.tag_` : Fine `POS` tags
- `.dep_` : Dependency labels
- `.head.text` : Syntactic head token
- `.ents` (`.text`,`.label`) Named Entities
- `.noun_chunks` Base noun phrases

#### 2.1 | Tokenised text

```python
[token.text for token in doc]
```

```
['Evidently',
 'someone',
 'with',
 'the',
 'authority',
 'to',
 'make',
 'decisions',
 'has',
 'arrived',
 '.']
 ```
 
 #### 2.2 | Lemmatised text

```python
[token.lemma_ for token in doc]
```

```
['evidently',
 'someone',
 'with',
 'the',
 'authority',
 'to',
 'make',
 'decision',
 'have',
 'arrive',
 '.']
 ```
 
 #### 2.3 | Parts-of-Speech (POS)
 
 - We have access to `coarser` and `finer` methods
 - `POS` tagging is used to assign tags to words, such as `nouns`, `verbs` etc

```python
# coarse POS tags
[token.pos_ for token in doc]
```

```
['ADV',
 'PRON',
 'ADP',
 'DET',
 'NOUN',
 'PART',
 'VERB',
 'NOUN',
 'AUX',
 'VERB',
 'PUNCT']
```

```python
# fine POS tags
[token.tag_ for token in doc]
```

```
['RB', 'NN', 'IN', 'DT', 'NN', 'TO', 'VB', 'NNS', 'VBZ', 'VBN', '.']
```

#### 2.3 | Syntatic dependencies

```python
# dependeny labels
# [token.dep_ for token in doc]
```

```
['advmod',
 'ROOT',
 'prep',
 'det',
 'pobj',
 'aux',
 'acl',
 'nsubj',
 'aux',
 'ccomp',
 'punct']
```

```python
# syntatic head tokens
[token.head.text for token in doc]
```

```
['someone',
 'someone',
 'someone',
 'authority',
 'with',
 'make',
 'authority',
 'arrived',
 'arrived',
 'make',
 'someone']
```

#### 2.5 | Named Entities
 
 ```python
 # Named Entities
 [(ent.text, ent.label_) for ent in doc.ents]
```
 
```
[]
```

#### 2.6 | Sentences

```python
[sent.text for sent in doc.sents]
```

```
['Evidently someone with the authority to make decisions has arrived.']
```

#### 2.7 | Base noun phrases

```python
[chunk.text for chunk in doc.noun_chunks]
```

```
['Evidently someone', 'the authority', 'decisions']
```

The content of the output actually depends on the `pipeline` content

### 3 | Pipeline Content

#### 3.1 | Check pipeline content

- Each model has a preloaded `pipeline` of NLP operations 
- Upon activating the `str` special method, these pipelines are activated
- We can visualise the pipeline step using `.pipe_names` & contents with `.pipeline`

```python
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)
```

```
['tok2vec', 'morphologizer', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
```

```python
print(nlp.pipeline)
```

```
 [('tok2vec', <spacy.pipeline.tok2vec.Tok2Vec object at 0x7fab27982670>), 
  ('tagger', <spacy.pipeline.tagger.Tagger object at 0x7fab2fe42130>), 
  ('parser', <spacy.pipeline.dep_parser.DependencyParser object at 0x7fab290957d0>), 
  ('attribute_ruler', <spacy.pipeline.attributeruler.AttributeRuler object at 0x7fab0100d280>), 
  ('lemmatizer', <spacy.lang.en.lemmatizer.EnglishLemmatizer object at 0x7fab01005230>), 
  ('ner', <spacy.pipeline.ner.EntityRecognizer object at 0x7fab2908eed0>)]
```
 
#### 3.2 | Built-in pipeline components

Depending on the `pipeline`, the content can slightly vary
 
- `tagger` Assign part-of-speech-tags
- `parser` Assign dependency labels
- `ner` Assign named entities
- `entity_linker` Assign knowledge base IDs to named entities. Should be added after the entity recognizer
- `entity_ruler` Assign named entities based on pattern rules and dictionaries
- `textcat` Assign text categories: exactly one category is predicted per document
- `textcat_multilabel` Assign text categories in a multi-label setting: zero, one or more labels per document
- `lemmatizer` Assign base forms to words using rules and lookups
- `trainable_lemmatizer` Assign base forms to words
- `morphologizer` Assign morphological features and coarse-grained POS tags
- `attribute_ruler` Assign token attribute mappings and rule-based exceptions
- `senter` Assign sentence boundaries
- `sentencizer` Add rule-based sentence segmentation without the dependency parse
- `tok2vec` Assign token-to-vector embeddings
- `transformer` Assign the tokens and outputs of a transformer model
 
#### 3.3 | Partially loading components
 
- If we don't want to load all components in the `pipeline`, we can remove them when activating the `str` special function
- For example, if we didn't want to load the `parser` & `ner` components of the `pipeline`
 
```python
doc = nlp(doc, disable=['parser', 'ner'])
```

#### 3.4 | Incorporating SpaCy pipeline in text cleaning 

Let's look at an example of how we can implement the `SpaCy` pipeline

- We will utilise `SpaCy` to tokenise `corpus` & utilise the `tokenised` & `lammatised` words (which we can access from `.lemma_`)

We need some extra libraries:

- We will also utilise `nltk`, which contains a `list` of so called `stop words`, which we will remove
- We will also utilise `string`, which contains a `list` of **punctuations**, anything in this list we will also remove

```python

# list of strings (our input corpus)
corpus = ['Evidently someone with the authority to make decisions has arrived.',
          'I think I smell the stench of your cologne, Agent Cooper.',
          'Smells like hubris.']

import spacy

# get english stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# get punctuations
import string
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
```

```
unprocessed: 
['Evidently someone with the authority to make decisions has arrived.', 'I think I smell the stench of your cologne, Agent Cooper.', 'Smells like hubris.']

processed: 
['evidently authority make decision arrive', 'think smell stench cologne agent cooper', 'smell like hubris']
```

### 3 | Spanning

Like strings, `spanning` is referred to selecting tokenised `text`, we can also use `:` to select multiple

```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('I like cats too.')
doc[1]
```

```
like
```

### 4 | Similarity & Word Vectors

#### 4.1 | Similarity

We can also use `SpaCy` to check the similarity between two input `strings` & compare them

```python
doc1 = nlp('I like cats too')
doc2 = nlp('I prefer cats over dogs')
doc3 = nlp('Tom and John went to the library')

print(doc1.similarity(doc2))
print(doc1.similarity(doc3))
```

```
0.585439442022622     
-0.012077887409129722
```

#### 4.2 | Word Vectors

- Words can be represented in **vector** format, `SpaCy`
- Select one of the `tokenised` words, and visualise the `vector` form

```python
print(doc1[2].vector) # word vector
print(f'normalised {doc1.vector_norm}') # enter sentence normalised value
```

```
[ 0.67897564  0.12110139 -0.6604409   0.0776536  -1.9418607   0.6940666
  1.19576     0.6619939   0.50233465  0.05248592 -0.19295473  1.2842398
  0.723488    0.64765275 -0.46113864 -0.12381127  0.31568098  0.5463978
 -1.376126   -0.2555286   0.6541349  -0.71979237 -0.14713815  0.79841524
 -0.9361088  -0.14297551  1.2247958   0.5354835  -0.5443332  -0.42707154
  0.55424935  0.8715273  -0.25182363 -1.5841036  -0.14750908 -0.65078586
 -0.17169908 -0.5729357  -0.13728389 -0.1380899   0.12419317 -0.25010562
  0.06765506  0.1825014  -0.6063776  -0.7749779   1.1444601   0.5669737
 -1.0870733  -0.39480096 -0.3147017  -0.10073815 -1.1867158  -1.7028933
  0.72262895  0.49310595  0.600273    0.16741765  0.57921666 -1.0983374
 -0.5445302  -1.3918273  -0.4013725   2.2526665   1.0592192  -0.45277885
 -0.6982554  -0.31604335  0.6993128   0.4112054  -0.02864948  0.13155304
 -0.77090514 -0.19796075 -0.30412257  1.1847382  -0.0515812  -0.14376783
  0.39952287  0.81643397  1.226792   -0.323776   -0.27826166 -1.7565243
 -0.2628545   0.1729832  -0.6630485  -0.49979892 -0.40806353 -1.7938248
 -0.56411684 -0.5260589   0.8754898   2.4937406   0.05924536 -1.4038779 ]
normalised 4.218990020889313
```
