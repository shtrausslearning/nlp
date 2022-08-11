
### [`SpaCy`](https://spacy.io) NLP library operations

#### 1 | Defining `model` & input `string`

Load statistical model `.load(model)`

```python
# nlp = spacy.load("ru_core_news_sm")
nlp = spacy.load("ru_core_news_lg")
# nlp = spacy.load('en_core_web_sm')
```

Define string `nlp(string)`

```python
doc = nlp('input string')
```

Upon defining `nlp()`, we have access to:

- `.text` : Tokens
- `.sents` Sentences
- `.pos_` : Coarse-grained `POS` tags
- `.tag_` : Fine-grained `POS` tags
- `.dep_` : Dependency labels
- `.head.text` : Syntactic head token (governor)
- `.ents` (`.text`,`.label`) Named Entities
- `.noun_chunks` Base noun phrases

```python
# [token.text for token in doc]
[token.pos_ for token in doc]
# [token.tag_ for token in doc]
# [token.dep_ for token in doc]
# [token.head.text for token in doc]
# [(ent.text, ent.label_) for ent in doc.ents]
# [sent.text for sent in doc.sents]
# [chunk.text for chunk in doc.noun_chunks]
```

### Pipeline

Each model has a preloaded `pipeline` of NLP operations upon 

#### Examples

```python
import spacy

# statistical model 
nlp = spacy.load("en_core_web_sm")

# load 
doc = nlp('Are you afraid of something?')

tokens = [token.text for token in doc]
print(f'Tokens (.text): {tokens}\n')
# Tokens (.text): ['Are', 'you', 'afraid', 'of', 'something', '?']

pos = [token.pos_ for token in doc] # Parts of speech
posf = [token.tag_ for token in doc] # Parts of speech
print(f'POS: {pos}',f'\n',f'POSF: {posf}\n')
# POS: ['AUX', 'PRON', 'ADJ', 'ADP', 'PRON', 'PUNCT'] 
# POSF: ['VBP', 'PRP', 'JJ', 'IN', 'NN', '.']

# Syntactic dependencies
depl = [token.dep_ for token in doc] # dependency labels
sht = [token.head.text for token in doc] # Syntactic head token (governor)
print(f'Syntatic dependencies: {depl}','\n',f'{sht}\n')
# Syntatic dependencies: ['ROOT', 'nsubj', 'acomp', 'prep', 'pobj', 'punct'] 
# ['Are', 'Are', 'Are', 'afraid', 'of', 'Are']

# Named Entities
doc = nlp("Larry Page founded Google")
ne = [(ent.text, ent.label_) for ent in doc.ents] # Text and label of named entity span
print(f'Named entities: {ne}\n')
# Named entities: [('Larry Page', 'PERSON')]

# Sentences
doc = nlp("This a sentence. This is another one.")
sents = [sent.text for sent in doc.sents]
print(f'Sentences: {sents}\n')
# Sentences: ['This a sentence.', 'This is another one.']

# Base Noun phrases
doc = nlp("I have a red car")
bn = [chunk.text for chunk in doc.noun_chunks] # doc.noun_chunks is a generator that yields spans
print(f'Base noun phrases: {bn}\n')
# Base noun phrases: ['I', 'a red car']

# Label Explanations
le1 = spacy.explain("RB") # 'adverb'
le2 = spacy.explain("GPE") # 'Countries, cities, states'
print(f'Label explanations: {le1},{le2}')
Label explanations: adverb,Countries, cities, states
```
