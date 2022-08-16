## Keras/Tensorflow Embedding Layer usage

Together with `GloVe` pretrained weights

### 1 | Word Embedding

- Way of representing `text`, where each word in the `vocabulary` is represented by a **real valued vector** in a high-dim space
- The vectors are learned in such a way that words that have similar meanings will have similar representation in the vector space (close in the vector space)
- This is a more expressive representation for text than more classical methods like **bag-of-words** (`BOW`), where relationships between words or tokens are ignored, or forced in bigram and trigram approaches
- The real valued `vector representation` for words can be learned (trainable weights) while training the neural network 
- We can do this in the Keras deep learning library using the `Embedding layer`

### 2 | Reading GloVe data

- Pretrained [`GloVe`](https://nlp.stanford.edu/projects/glove/), let's choose `glove.6B.100d.txt`
- It was trained on a dataset of one billion tokens (words) with a vocabulary of 400 thousand words. 
- There are a few different embedding vector sizes, including 50, 100, 200 and 300 dimensions.

We can check the format in the file `glove.6B.100d.txt`, for each word we have a **vector representation**:

```
the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131 -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594 -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223 -0.29871 -0.15749 -0.34758 -0.045637 -0.44251 0.18785 0.0027849 -0.18411 -0.11514 -0.78581
, 0.013441 0.23682 -0.16899 0.40951 0.63812 0.47709 -0.42852 -0.55641 -0.364 -0.23938 0.13001 -0.063734 -0.39575 -0.48162 0.23291 0.090201 -0.13324 0.078639 -0.41634 -0.15428 0.10068 0.48891 0.31226 -0.1252 -0.037512 -1.5179 0.12612 -0.02442 -0.042961 -0.28351 3.5416 -0.11956 -0.014533 -0.1499 0.21864 -0.33412 -0.13872 0.31806 0.70358 0.44858 -0.080262 0.63003 0.32111 -0.46765 0.22786 0.36034 -0.37818 -0.56657 0.044691 0.30392
. 0.15164 0.30177 -0.16763 0.17684 0.31719 0.33973 -0.43478 -0.31086 -0.44999 -0.29486 0.16608 0.11963 -0.41328 -0.42353 0.59868 0.28825 -0.11547 -0.041848 -0.67989 -0.25063 0.18472 0.086876 0.46582 0.015035 0.043474 -1.4671 -0.30384 -0.023441 0.30589 -0.21785 3.746 0.0042284 -0.18436 -0.46209 0.098329 -0.11907 0.23919 0.1161 0.41705 0.056763 -6.3681e-05 0.068987 0.087939 -0.10285 -0.13931 0.22314 -0.080803 -0.35652 0.016413 0.10216
of 0.70853 0.57088 -0.4716 0.18048 0.54449 0.72603 0.18157 -0.52393 0.10381 -0.17566 0.078852 -0.36216 -0.11829 -0.83336 0.11917 -0.16605 0.061555 -0.012719 -0.56623 0.013616 0.22851 -0.14396 -0.067549 -0.38157 -0.23698 -1.7037 -0.86692 -0.26704 -0.2589 0.1767 3.8676 -0.1613 -0.13273 -0.68881 0.18444 0.0052464 -0.33874 -0.078956 0.24185 0.36576 -0.34727 0.28483 0.075693 -0.062178 -0.38988 0.22902 -0.21617 -0.22562 -0.093918 -0.80375
to 0.68047 -0.039263 0.30186 -0.17792 0.42962 0.032246 -0.41376 0.13228 -0.29847 -0.085253 0.17118 0.22419 -0.10046 -0.43653 0.33418 0.67846 0.057204 -0.34448 -0.42785 -0.43275 0.55963 0.10032 0.18677 -0.26854 0.037334 -2.0932 0.22171 -0.39868 0.20912 -0.55725 3.8826 0.47466 -0.95658 -0.37788 0.20869 -0.32752 0.12751 0.088359 0.16351 -0.21634 -0.094375 0.018324 0.21048 -0.03088 -0.19722 0.082279 -0.09434 -0.073297 -0.064699 -0.26044
and 0.26818 0.14346 -0.27877 0.016257 0.11384 0.69923 -0.51332 -0.47368 -0.33075 -0.13834 0.2702 0.30938 -0.45012 -0.4127 -0.09932 0.038085 0.029749 0.10076 -0.25058 -0.51818 0.34558 0.44922 0.48791 -0.080866 -0.10121 -1.3777 -0.10866 -0.23201 0.012839 -0.46508 3.8463 0.31362 0.13643 -0.52244 0.3302 0.33707 -0.35601 0.32431 0.12041 0.3512 -0.069043 0.36885 0.25168 -0.24517 0.25381 0.1367 -0.31178 -0.6321 -0.25028 -0.38097
```

Load the whole `GloVe` embeddings into memory

```python
embeddings_index = {}

f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs

f.close()
print(f'Loaded {len(embeddings_index)} word vectors.')
```

```
Loaded 400000 word vectors.
```

### 3 | Define Data

Let's define a **corpus** `docs`, which will have a corresponding label `label`

```python
# Define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
		
# Define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])
```

Tokenise the corpus, `docs`

- Set tokeniser, `Toknizer()`
- Fit onto corpus `docs` 

We will obtain encoded `docs`, `encoded_docs` & its corresponding dictionary `tokeniser.word_index`

```python
# Prepare tokenizer
tokeniser = Tokenizer()
tokeniser.fit_on_texts(docs)
vocab_size = len(tokeniser.word_index) + 1

# Integer encode the documents
encoded_docs = tokeniser.texts_to_sequences(docs)
print(f'encoded documents: {encoded_docs}')`
```

```
encoded documents: [[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]
```

The dictionary for the tokenised `doc`, `tokeniser.word_index`

```python
tokeniser.word_index
```

```python
{'work': 1,
 'done': 2,
 'good': 3,
 'effort': 4,
 'poor': 5,
 'well': 6,
 'great': 7,
 'nice': 8,
 'excellent': 9,
 'weak': 10,
 'not': 11,
 'could': 12,
 'have': 13,
 'better': 14}
```

Apply padding to each tokenised `list`, setting the maximum length `maxlen` to 4 

```python
# Pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(f'padded documents:\n {padded_docs}')
```

```
padded documents:
 [[ 6  2  0  0]
 [ 3  1  0  0]
 [ 7  4  0  0]
 [ 8  1  0  0]
 [ 9  0  0  0]
 [10  0  0  0]
 [ 5  4  0  0]
 [11  3  0  0]
 [ 5  1  0  0]
 [12 13  2 14]]
 ```

### 4 | Find Overlap between dictionary & `GloVe` vocabulay

Find the embedding vectors in `GloVe` for each of the words in our dictionary `tokeniser.word_index`

```python
# Create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))

# Cycle through all words in tokenised dictionary
for word, i in c.items():	
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
```

### 5 | Define a Model

Binary classification model w/ `embedding` layer

```python
# Define the model
model = Sequential()

# Use GloVe weights (non trainable)
emb_layer = Embedding(input_dim=vocab_size,         # Input into embedding layer
                      output_dim = 100,             # Output out of embedding layer
                      weights=[embedding_matrix],   # Custom weights
                      input_length=max_length,      # Input length             
                      trainable=False)              # Trainable weights in layer

# Trainable Embedding Layer
# emb_layer = Embedding(input_dim=vocab_size,output_dim = 8, 
# 					            input_length=max_length,trainable=True)

model.add(emb_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# summarize the model
print(model.summary())
```

```

```
