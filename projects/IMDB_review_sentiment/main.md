
## IMDB Review Sentiment 

- Load libraries, we'll be using `tensorflow` & `keras`

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd 
import os
```

- Load data from `ibdm.load_data` (`tf.keras.datasets`)

The method contains already preprocessed `text` data:

- The dataset contains multiple `text` data, each text has been converted to string & `tokenised`
- For all unique `tokens`, we have a `dictionary` of words, which we can limit using `num_words`
- We have a correspoding numerical representation for each `text` review

```python

- For starters we want the top 10,000 words that most frequently occurred in the training dataset
- We have 2500 reviews in both the `training` & `test` datasets

''' Load Data '''

((XTr,YTr),(XTe,YTe)) = imdb.load_data(num_words=10000) # tuple of numpy arrays

print("The length of the Training Dataset is ", len(XTr),' reviews')
print("The length of the Testing Dataset is ", len(XTe),' reviews')
```

```
The length of the Training Dataset is  25000  reviews
The length of the Testing Dataset is  25000  reviews
```

- Let's print one of the reviews in the training data `Xtr[0]`, the length of this particular review is 218 
- Each review has been converted into `numerical` format & has its own unique review length

```python

''' Print Example '''
# numeric representation of the words (1D arrays)

print(f'\nReview:\n{XTr[0]}')
print(f'\nLength of Review: {len(XTr[0])}')
```

```
# Review:
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 
# 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 
# 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 
# 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 
# 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 
# 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 
# 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 
# 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 
# 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 
# 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 
# 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224,
# 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 
# 32, 15, 16, 5345, 19, 178, 32]
# Length of Review: 218
```

- For `sentiment` classification, we have only two classes, [0,1]

#### Target Variable

```python
''' Target Variable '''
# Binary Classification
pd.Series(YTr).unique()
```

```
[0,1]
```

- If we wanted to convert the `numerical` representation back to `text` representation, `to_text`

```python
# convert this numeric review to a text review

def to_text(review):

    word_idx = imdb.get_word_index() # Dict with the words mapped to numbers 
    idx_word = dict([value,key] for (key,value) in word_idx.items()) # invert dictionary

    # if there is no matching index value in the dict, -> ? is used
    # indices are offset by three because 0, 1, and 2 are reserved 
    # indices for “padding”, “start of sequence” and “unknown”
    actual_review = ' '.join([idx_word.get(ii-3,'?') for ii in review])

    print(f'\nReview:\n{actual_review}')
    print(f'\nLength of Review: {len(actual_review.split())}')
    
# to_text(XTr[0])
```

#### Padding

- Each review, is currently in numerical form `int`, stored in a `list`
- Each `list` is of different length, so we'll need to pad each `list` to a common length 
- `sequence.pad_sequences` can be used to pad a 1D array (which contains the lists), let's set the review maximum to 500
- If a review is greater than 500 (length of the list), the list will be cut off & if it is smaller, 0's will be added to the end of the list 

```python
''' Perform Padding '''
# Convert 1D array -> 2D matrix
# Embedding layer in our model will accept a 2-D tensor

X_train = sequence.pad_sequences(XTr,maxlen=500)
X_test = sequence.pad_sequences(XTe,maxlen=500)

print(f'Shape After Padding (X_train) : {X_train.shape}')
print(f'Shape After Padding (X_test)  : {X_test.shape}')
```

```
Shape After Padding (X_train) : (25000, 500)
Shape After Padding (X_test)  : (25000, 500)
```

#### Model 

- Embedding matrix that gets trained and is shared among all the inputs and RNN cells
- Input to the embedding layer will be a 2D matrix of size (batch_size, maxlen of the sentences), and we already defined the maximum length of the sentences to be 500
- The embedding matrix is of size (vocab_size, K), vocab_size (10,000) & K (64) (here)
- The output from this embedding layer size (batch_size, maxlen of the sentence, K)
- When we defined the embedding layer in the above code, we passed the parameter (10,000, 64), meaning the vocab_size (10,000) & the representation of each word must be a vect of length 64

```python
''' Sequential Model '''
# Binary Classifier 

model = Sequential()
model.add(Embedding(10000,64))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

# Compile Model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
```


''' Callbacks '''
# During training, save model & weights if there
# has been an improvement, stop early 

```python
cb1 = ModelCheckpoint("best_model.h5",
                      monitor='val_loss', 
                      verbose=0, 
                      save_best_only=True, 
                      save_weights_only=False)

cb2 = EarlyStopping(monitor='val_acc',
                   patience=1)
```

```python
''' Fit Model '''
# Train model on training dataset and 
# use validation of 0.2, run for 10 epochs
# using bactch size of 128

hist = model.fit(X_train,YTr,
                 validation_split=0.2,
                 epochs=10,
                 batch_size=128,
                 callbacks=[cb1,cb2])
```


```
# Epoch 1/10
# 157/157 [==============================] - 60s 374ms/step - loss: 0.5817 - acc: 0.6905 - val_loss: 0.4157 - val_acc: 0.8282
# Epoch 2/10
# 157/157 [==============================] - 57s 365ms/step - loss: 0.3630 - acc: 0.8530 - val_loss: 0.3966 - val_acc: 0.8374
# Epoch 3/10
# 157/157 [==============================] - 57s 360ms/step - loss: 0.2498 - acc: 0.9026 - val_loss: 0.3663 - val_acc: 0.8578
# Epoch 4/10
# 157/157 [==============================] - 58s 368ms/step - loss: 0.1892 - acc: 0.9288 - val_loss: 0.4033 - val_acc: 0.8594
# Epoch 5/10
# 157/157 [==============================] - 57s 366ms/step - loss: 0.1172 - acc: 0.9610 - val_loss: 0.4450 - val_acc: 0.8586
# Epoch 6/10
# 157/157 [==============================] - 58s 367ms/step - loss: 0.0729 - acc: 0.9769 - val_loss: 0.4707 - val_acc: 0.8412
# Epoch 7/10
# 157/157 [==============================] - 56s 359ms/step - loss: 0.0471 - acc: 0.9861 - val_loss: 0.5152 - val_acc: 0.8406
```

### Evaluation 

```python
''' Evaluation on Test Set '''
# Having trained our model, let's test how well it performs on onseen data

model.evaluate(X_test,Yt)

# > 782/782 [==============================] - 16s 21ms/step - loss: 0.4679 - acc: 0.8047
# > [0.4679102599620819, 0.8046799898147583]

''' Create a custom review & use model to sentiment '''
# When we want to utilise the model to conduct a sentiment analysis
# we can load the model, h5 format

# It can be used to reconstruct the model identically
reconstructed_model = load_model("best_model.h5")

def pred_imdb(sentence,model):

    inp = []
    print(f'Input sentence: \n{sentence}')

    # get dict mapper
    dict_word = imdb.get_word_index() # get dict mapper

    # convert words to int
    for word in sentence.split():
        if word in dict_word.keys():
            inp.append(dict_word[word])
        else:
            inp.append(1)

    print(f'Converted sentence: \n{inp}')

    # Perform padding
    final_input = pp.sequence.pad_sequences([inp],maxlen=500)

    # Finally predict the sentiment
    return model.predict(final_input)
    
# Predict if sent is a good or bad review
sent = "This movie is really bad"
pred_imdb(sent,reconstructed_model)
