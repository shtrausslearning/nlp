## Change words to base word state

### 1 | Stemming 

Change words to normal form, by **dropping the ending** by using the rules of the language

#### nltk `stem`

- Doesn't work for russian language
- Define stemmer, `SnowballStemmer(lang)` and use `.stem`

```python
from nltk.stem.snowball import SnowballStemmer

stem_eng = SnowballStemmer(language='english').stem('running')
stem_rus = SnowballStemmer(language='russian').stem('бежать')
print(stem_eng,stem_rus)
```

```
run бежа
```

#### 2 | Lemmatisation

Change words to a **starting morphological form** (using dictionary & grammar of the language)

#### Pymorphy2 `MorphAnalyzer()`

- `MorphAnalyzer` & `.parse('word')`, get the normal form `.normal_form`
- **`MorphAnalyzer`** doesn't take into account the context of the sentence, thus might show wrong result

```python

from pymorphy2 import MorphAnalyzer

pymorphy = MorphAnalyzer()
output = pymorphy.parse('бежал')
print(output)
print(output[0].normal_form)
```

```
[Parse(word='бежал', tag=OpencorporaTag('VERB,perf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 1),)), Parse(word='бежал', tag=OpencorporaTag('VERB,impf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 49),))]
бежать
```

```python
def lemPymorphy(tokens):
    lemms = [pymorphy.parse(token)[0].normal_form for token in tokens]
    return lemms

lemPymorphy(['бегут','бежал',',бежите'])
```

```
['бежать', 'бежать', ',бежать']
```

```python
pymorphy = MorphAnalyzer()
pymorphy.normal_forms('На заводе стали увидел виды стали')
```

```
['на заводе стали увидел виды стать', 'на заводе стали увидел виды сталь']
```

```python
lemPymorphy(['На','заводе','стали','увидел','виды','стали'])
```

```
['на', 'завод', 'стать', 'увидеть', 'вид', 'стать']
```

#### Pymystem3 `Mystem`

- Yandex `mystem3` bypasses the above issues & takes into account the **context of the sentence** using statistics and rules

```python
from pymystem3 import Mystem

mystem = Mystem()
def lemPymystem(text):
    lemms = [token for token in mystem.lemmatize(text) if token != ' '][:-1]
    return lemms

lemPymystem('бегал бежал ')
```

```
['бегать', 'бежать']
```

## Creating Features for ML modeling

- One-Hot Encoding Features (preprocessing.OneHotEncoder)
- Bag-of-Words Feature (feature_extraction.text.CountVectorizer)
- TF-IDF Feature (feature_extraction.text.TfidfVectorizer)

### 1 | One-Hot Encoding

```python

''' One-Hot Encoding Features'''
# Vertical direction -> number of words in the list (sentences are treated as words eg. "NLP today") 
# Horizontal direction -> unique name in list 
# Value represents the location where the word occurs (eg. awesome -> 3rd & last string, 1 value @ its identifier location -> 3rd column (3rd in categories_))

import numpy as np
from sklearn.preprocessing import OneHotEncoder

words = np.array(['NLP','is','awesome','eh','NLP today','awesome'])

encoder = OneHotEncoder(sparse=False)
encoder.fit_transform(words[:,None])
```

```
array([[1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.]])
```

```python
encoder.categories_
```

```
[array(['NLP', 'NLP today', 'awesome', 'eh', 'is'], dtype='<U9')]
```

### 2 | Bag of Words

```python
''' Bag of Words Features'''
# Vertical direction -> number of sentences in corpus
# Horizontal diction -> Vocabulary ID 
# Value represents the number of words found in the sentence

from sklearn.feature_extraction.text import CountVectorizer

# corpus

corpus = [
    'Girl likes cat Tom',
    'Who likes the cat?',
    'Tom is a quiet cat'
]

vectoriser = CountVectorizer()
vectors = vectoriser.fit_transform(corpus)

vectors.toarray()
```

```
array([[1, 1, 0, 1, 0, 0, 1, 0],
       [1, 0, 0, 1, 0, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1, 0]])
```

```python
# vocabulary IDs
vectoriser.vocabulary_
```

```
{'girl': 1,
 'likes': 3,
 'cat': 0,
 'tom': 6,
 'who': 7,
 'the': 5,
 'is': 2,
 'quiet': 4}
```

### 3 | TF-IDF 

```python
''' TF-IDF features '''
# corpus -> 3 lines -> 3 vectors
# length of vectors equal to the vocabulary length

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'Girl likes cat Tom',
    'Who likes the cat?',
    'Tom is a quiet cat'
]

vectoriser = TfidfVectorizer()
vectors = vectoriser.fit_transform(corpus)
vectors.toarray()
```

```
array([[0.37311881, 0.63174505, 0.        , 0.4804584 , 0.        ,
        0.        , 0.4804584 , 0.        ],
       [0.34520502, 0.        , 0.        , 0.44451431, 0.        ,
        0.5844829 , 0.        , 0.5844829 ],
       [0.34520502, 0.        , 0.5844829 , 0.        , 0.5844829 ,
        0.        , 0.44451431, 0.        ]])
```

```python
# Vocabulary Identifier
vectoriser.vocabulary_
```

```
{'girl': 1,
 'likes': 3,
 'cat': 0,
 'tom': 6,
 'who': 7,
 'the': 5,
 'is': 2,
 'quiet': 4}
```
