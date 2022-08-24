
### 1 | Tokenisation

- Tokenization breaks the raw text into words, sentences called tokens
- These tokens help in understanding the context or developing the model for the NLP
  - If the text is split into words using some separation technique it is called `word tokenization`
  - For sentences is called `sentence tokenization`

```python
paragraph = "write paragaraph here to convert into tokens."
```

#### 1.1 | NLTK module

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

#### 1.2 | SpaCy module

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

#### 1.3 | Keras/TF module

```python
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(paragraph)
```

#### 1.4 | Gensim module

```python
from gensim.summarization.textcleaner import split_sentences
split_sentences(paragraph)

from gensim.utils import tokenize
list(tokenize(paragraph))
```

### 2 | Feature Generation

#### 2.1 | Bag of Words `BOW`

- Bag of Words model is used to preprocess the text by converting it into a bag of words, which keeps a count of the total occurrences of most frequently used words
- counters = List of stences after pre processing like tokenization, stemming/lemmatization, stopwords

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(counters).toarray()
```

#### 2.2 | BOW `ngrams`

N-gram Language Model:  An N-gram is a sequence of N tokens (or words)

- `1-gram` (or unigram) is a one-word sequence.the unigrams would simply be: “I”, “love”, “reading”, “blogs”, “about”, “data”, “science”, “on”, “Analytics”, “Vidhya”
- `2-gram` (or bigram) is a two-word sequence of words, like “I love”, “love reading”, or “Analytics Vidhya”
- `3-gram` (or trigram) is a three-word sequence of words like “I love reading”, “about data science” or “on Analytics Vidhya”


#### 2.3 | TF-IDF features

Term Frequency-Inverse Document Frequency `TF-IDF`

- Numerical statistic that is intended to reflect **how important a word is** to a document in a collection or corpus
- `T.F`  No of rep of words in setence/No of words in sentence
- `IDF` No of sentences / No of sentences containing words

```python
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(counters).toarray()
```

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
