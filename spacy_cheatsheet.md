
![](https://i.imgur.com/rRZBFnz.png)

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

#### Tokenised text

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
 
 #### Lemmatised text

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
 
 #### Parts-of-Speech (POS)

```python
# coarse pos tags
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
# fine pos tags
[token.tag_ for token in doc]
```

```
['RB', 'NN', 'IN', 'DT', 'NN', 'TO', 'VB', 'NNS', 'VBZ', 'VBN', '.']
```

#### Syntatic dependencies

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

#### Named Entities
 
 ```python
 # Named Entities
 [(ent.text, ent.label_) for ent in doc.ents]
```
 
```
[]
```

#### Sentences

```python
[sent.text for sent in doc.sents]
```

```
['Evidently someone with the authority to make decisions has arrived.']
```

#### Base noun phrases

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