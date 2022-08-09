
### 1 | CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?' ]

vectoriser = CountVectorizer()
X = vectoriser.fit_transform(corpus)
print(X.shape)
print(X.toarray())
vocab = vectoriser.get_feature_names_out()
print(vocab,len(vocab),'\n')
```

```
(4, 9)
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this'] 9 
```

```python
vectorizer2 = CountVectorizer(analyzer='word', 
                              ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
vocab = vectorizer2.get_feature_names_out()
print(X2.shape)
print(X2.toarray())
print(vocab,len(vocab),'\n')
```

```
(4, 13)
[[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]
['and this' 'document is' 'first document' 'is the' 'is this'
 'second document' 'the first' 'the second' 'the third' 'third one'
 'this document' 'this is' 'this the'] 13 
 ```
 
