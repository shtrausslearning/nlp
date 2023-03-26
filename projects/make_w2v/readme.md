Two quick tasks that utilise **embedding layers** are commonly utilised in NLP to learn `word embeddings`

`CBOW approach`

- The CBOW model architecture is a type of neural network
- It's usually a shallow neural network (embedding,ll) which aims to predict a target word, based on its **context of words**
- The model aims to **predict the missing word in the centre**, given a **set of words of its surrounding** (eg. +/ n words)
- The CBOW model is trained by feeding it with large amounts of text data, and it learns to associate words that appear in similar contexts
- This allows the model to generate accurate word embeddings, which can be used for various NLP tasks 

The context pair vector can be generated quite easiliy:

```python
# loop through all possible cases 
for i in range(window,len(tokens) - window):
    
    context = []
    
    # words to the left
    for j in range(-window,0):
        context.append(tokens[i+j])
    
    # words to the right
    for j in range(1,window+1):
        context.append(tokens[i+j])
        
    context_pairs.append((context,tokens[i]))
```

context, target pairs

(['Today', 'is', 'good', 'day'], 'a')
(['is', 'a', 'day', 'for'], 'good')
(['a', 'good', 'for', 'taking'], 'day')
(['good', 'day', 'taking', 'a'], 'for')
(['day', 'for', 'a', 'walk'], 'taking')

for pytorch; context, target word tensors

tensor([0, 1, 4, 6]) tensor(5)
tensor([1, 5, 6, 7]) tensor(4)
tensor([5, 4, 7, 3]) tensor(6)
tensor([4, 6, 3, 5]) tensor(7)
tensor([6, 7, 5, 2]) tensor(3)

`Skip-gram approach`

- A skipgram model is also a neural network
- Out of the two approaches, the SG approach is a little less intuitive
- The skipgram model **predicts the context words**, which **surround the target word** 
- It takes a target word as input and tries to predict the probability of occurrence of surrounding words in a fixed window size
- It learns the meaning and relationships b/w dictionary words, based on their context in the training data

Tensorflow has a module which can simplify the generation of skipgrams:

```python
from tensorflow.keras.preprocessing.sequence import skipgrams

# Enumerate over tokenised text
for i, doc in enumerate(tokeniser.texts_to_sequences(corpus)):

    data, labels = skipgrams(sequence=doc, 
                             vocabulary_size=vocab_size, 
                             window_size=1,
                             shuffle=False)

    x = [np.array(x) for x in zip(*data)]
    y = np.array(labels, dtype=np.int32)

```

[[1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 1], [1, 6], [1, 7], [7, 1], [7, 8], [8, 7], [8, 6], [7, 6], [3, 1], [7, 7], [4, 3], [5, 2], [2, 4], [1, 4], [3, 1], [5, 5], [4, 2], [6, 2], [6, 2], [1, 5], [1, 1], [2, 2]] [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
