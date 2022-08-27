
## 1 | Tokenisation

There are quite a few modules besides `.split()` that we can use to tokenise & proprocess text

- `Tokenization` breaks the raw text into words, sentences called `tokens`
- These tokens help in understanding the context or developing the model for the NLP
  - If the text is split into words using some separation technique it is called `word tokenization`
  - For sentences is called `sentence tokenization`

### 1.1 | NLTK module

`NLTK` module has a few `tokenisers`

- `sent_tokenize`, `word_tokenize`
- `TreebankWordTokenizer`
- `WordPunctTokenizer`
- `RegexpTokenizer`

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

paragraph = "write paragaraph here to convert into tokens."

nltk.download('punkt')

sentences = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(paragraph)

print(sentences)
print(words)
```

```
['write paragaraph here to convert into tokens.']
['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens', '.']
```
  
```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(text)
```

```
['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens', '.']
```

```python
from nltk.tokenize import WordPunctTokenizer
  
tokenizer = WordPunctTokenizer()
tokenizer.tokenize("Let's see how it's working.")
```

```
['Let', "'", 's', 'see', 'how', 'it', "'", 's', 'working', '.']
```

```python
from nltk.tokenize import RegexpTokenizer
  
tokenizer = RegexpTokenizer("[\w']+")
text = "Let's see how it's working."
tokenizer.tokenize(text)
```

```
["Let's", 'see', 'how', "it's", 'working']
```

### 1.2 | SpaCy module

We can use `SpaCy`'s `English()` module & add a `sentencizer` to the existing pipeline

```python
from spacy.lang.en import English
nlp = English()
sbd = nlp.create_pipe('sentencizer')
nlp.add_pipe(sbd)

doc = nlp(paragraph)
[sent for sent in doc.sents]

nlp = English()
doc = nlp(paragraph)
[word for word in doc]
```

We can load other models eg. `en_core_web_sm`, which already contains a `sentencizer` in the pipeline

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
    
tokens = [token.text for token in doc]
sentences = [token.text for token in doc.sents]
print(tokens)
print(sentences)
```

```
['Let', "'s", 'see', 'how', 'it', "'s", 'working', '.']
["Let's see how it's working."]
```

### 1.3 | Keras/TF module

```python
from keras.preprocessing.text import text_to_word_sequence

paragraph = "write paragaraph here to convert into tokens."

text_to_word_sequence(paragraph)
```

```
['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens']
```

### 1.4 | Gensim module

```python
from gensim.utils import tokenize

paragraph = "write paragaraph here to convert into tokens."
list(tokenize(paragraph))
```

```
['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens']
```

### 1.5 | Pymorphy2 module

```python
from pymorphy2.tokenizers import simple_word_tokenize 
paragraph = "write paragaraph here to convert into tokens."
simple_word_tokenize(paragraph)
```

```
['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens', '.']
```

### 1.6 | Razdel

```python
import razdel

paragraph = "write paragaraph here to convert into tokens."

sentences = [sentence.text for sentence in razdel.sentenize(paragraph)]
tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
print(sentences)
print(tokens)
```

```
['write paragaraph here to convert into tokens.']
[['write', 'paragaraph', 'here', 'to', 'convert', 'into', 'tokens', '.']]
```

<br>

## 2 | Feature Generation

- There are various way to generate **features** from `text` for **machine learning** application that the model will be able to interpret
- **Feature matrices** which have both `numeric` and `categorical` features, depending on the model need to be converted to numerical form

### 2.1 | Label Encoder 

```python
from sklearn.preprocessing import LabelEncoder

paragraph = "write paragaraph here to convert into tokens."
lst_paragraph = paragraph.split(' ')

label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(lst_paragraph)
corpus_encoded
```

```
array([6, 3, 1, 4, 0, 2, 5])
```

### 2.2 | One Hot Encoder (OHE)

`OHE` works on both `numerical` & `string` features 

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

paragraph = "write paragaraph here to convert into tokens."
lst_paragraph = paragraph.split(' ')
np_paragraph = np.array(lst_paragraph)
print(np_paragraph)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(np_paragraph[:,None])
```

```
['write' 'paragaraph' 'here' 'to' 'convert' 'into' 'tokens.']
array([[0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.]])
```

### 2.3 | Bag of Words (BoW)

- Bag of Words model is used to preprocess the text by converting it into a bag of words, which keeps a count of the total occurrences of most frequently used words
- counters = List of stences after pre processing like tokenization, stemming/lemmatization, stopwords

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(counters).toarray()
```

### 2.4 | BoW ngrams

N-gram Language Model:  An N-gram is a sequence of N tokens (or words)

- `1-gram` (unigram) is a one-word sequence.the unigrams would simply be: “I”, “love”, “reading”, “blogs”, “about”, “data”, “science”, “on”, “Analytics”, “Vidhya”
- `2-gram` (bigram) is a two-word sequence of words, like “I love”, “love reading”, or “Analytics Vidhya”
- `3-gram` (trigram) is a three-word sequence of words like “I love reading”, “about data science” or “on Analytics Vidhya”


### 2.5 | TF-IDF features

Term Frequency-Inverse Document Frequency `TF-IDF`

Numerical statistic that is intended to reflect **how important a word is** to a document in a collection or corpus

- `T.F`  No of rep of words in setence/No of words in sentence
- `IDF` No of sentences / No of sentences containing words

```python
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(counters).toarray()
```

<br>

### 3 | Stemming & Lemmatisation

### 3.1 | Stemming

- From `stemming` we will process of getting the root form of a word 
- We would create the stem words by removing the prefix of suffix of a word
- So, stemming a word may not result in actual words

```python
paragraph = ""
```

#### NLTK module

```python
from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from nltk import word_tokenize
stem = PorterStemmer()

sentence = sent_tokenize(paragraph)[1]
words = word_tokenize(sentence)
[stem.stem(word) for word in words]
```

### 3.2 | Lemmatization

- `Lemmatization` do the same, only diff is that lemmatization ensures that root word belongs to the language

#### NLTK module 

```python
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

sentence = sent_tokenize(paragraph)[1]
words = word_tokenize(sentence)
[lemma.lemmatize(word) for word in words]
```

#### SpaCy module

```python
import spacy as spac
sp = spac.load('en_core_web_sm')
ch = sp(u'warning warned')
for x in ch:
    print(ch.lemma_)
```
