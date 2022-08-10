
Refernce library [`SpaCy`](https://spacy.io)

```python
import spacy

# statistical model 
nlp = spacy.load("en_core_web_sm")

# load strings
doc = nlp('Are you afraid of something?')

tokens = [token.text for token in doc]
print(tokens)
```

```python
# Parts of Speech
pos = [token.pos_ for token in doc] # Parts of speech
posf = [token.tag_ for token in doc] # Parts of speech
print(pos,'\n',posf)
```

```python
# Syntactic dependencies
depl = [token.dep_ for token in doc] # dependency labels
sht = [token.head.text for token in doc] # Syntactic head token (governor)
print(depl,'\n',sht)
```

```python
# Named Entities
doc = nlp("Larry Page founded Google")
ne = [(ent.text, ent.label_) for ent in doc.ents] # Text and label of named entity span
print(ne)
```

```python
# Sentences
doc = nlp("This a sentence. This is another one.")
sents = [sent.text for sent in doc.sents]
print(sents)
```

```python
# Base Noun phrases
doc = nlp("I have a red car")
bn = [chunk.text for chunk in doc.noun_chunks] # doc.noun_chunks is a generator that yields spans
print(bn)
```

```python
# Label Explanations
le1 = spacy.explain("RB") # 'adverb'
le2 = spacy.explain("GPE") # 'Countries, cities, states'
print(le1,le2)
```
