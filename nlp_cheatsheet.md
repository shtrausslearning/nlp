
### 1 | Tokenisation

- Tokenization breaks the raw text into words, sentences called tokens
- These tokens help in understanding the context or developing the model for the NLP
  - If the text is split into words using some separation technique it is called `word tokenization`
  - For sentences is called `sentence tokenization`

#### 1.1 | NLTK module

```python
import nltk
nltk.download('punkt')
paragraph = "write paragaraph here to convert into tokens."

sentences = nltk.sent_tokenize(paragraph)

words = nltk.word_tokenize(paragraph)
```

#### 1.2 | SpaCy module

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
