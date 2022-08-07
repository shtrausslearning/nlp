
## Text Classification 

- <code>IMDB</code> **sentiment classification** dataset (unprocessed variant) 
- Common <code>Binary</code> classification problem for text sentiment 

### 1 | The Dataset

- Unlike <code>text_classification.md</code>, the data from <code>keras.datasets.imdb.load_data</code> has been cleaned and returns  

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Get dataset
max_features = 10000  # Only consider the top 10000 words
maxlen = 200          # Only consider the first 200 words of each movie review

from tensorflow.keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
```

```
25000 Training sequences
25000 Validation sequences
```

- list/array of <code>tokenised</code> words, each review array can have a different length, so we need to <code>pad</code>

```python

print(f'Example review from {X_train.shape}')
print(len(X_train[0]))
print(X_train[0])
print(f'Min dict-index: {min(X_train[0])},Max dict-index: {max(X_train[0])} from a total of {max_features}')
```

```
Example review from (25000,)
218
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
Min dict-index: 1,Max dict-index: 7486 from a total of 10000
```

- <code>Pad</code> sequences, restrict maximum **array length to 200**, so we have a consistent matrix input into the model

```python

X_train = pad_sequences(X_train,maxlen=maxlen)
X_val = pad_sequences(X_val,maxlen=maxlen)

print(f'\nPadded Matrix: \n{X_train.shape}')
print(f'Example: {X_train[0].shape}')
print(X_train[0])
```

```
Padded Matrix: 
(25000, 200)
Example: (200,)
[   5   25  100   43  838  112   50  670    2    9   35  480  284    5
  150    4  172  112  167    2  336  385   39    4  172 4536 1111   17
  546   38   13  447    4  192   50   16    6  147 2025   19   14   22
    4 1920 4613  469    4   22   71   87   12   16   43  530   38   76
   15   13 1247    4   22   17  515   17   12   16  626   18    2    5
   62  386   12    8  316    8  106    5    4 2223 5244   16  480   66
 3785   33    4  130   12   16   38  619    5   25  124   51   36  135
   48   25 1415   33    6   22   12  215   28   77   52    5   14  407
   16   82    2    8    4  107  117 5952   15  256    4    2    7 3766
    5  723   36   71   43  530  476   26  400  317   46    7    4    2
 1029   13  104   88    4  381   15  297   98   32 2071   56   26  141
    6  194 7486   18    4  226   22   21  134  476   26  480    5  144
   30 5535   18   51   36   28  224   92   25  104    4  226   65   16
   38 1334   88   12   16  283    5   16 4472  113  103   32   15   16
 5345   19  178   32]
```

- <code>Tokenization</code> is the process of splitting a stream of language into individual tokens <code>Tokenizer</code>, <code>pad_sequences</code> 
- <code>Vectorization</code> is the process of converting string data into a numerical representation <code>TextVectorization</code>

### 2 | Model Architecture

```python

''' Model Architecture '''

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)

# Two LSTM layers (forward only)
# x = layers.LSTM(64,return_sequences=True)(x)
x = layers.LSTM(64)(x)

# Add a classifier layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile Model
model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics=["accuracy"])
```

```
Model: "model_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_18 (InputLayer)        [(None, None)]            0         
_________________________________________________________________
embedding_17 (Embedding)     (None, None, 128)         1280000   
_________________________________________________________________
lstm_17 (LSTM)               (None, 64)                49408     
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 65        
=================================================================
Total params: 1,329,473
Trainable params: 1,329,473
Non-trainable params: 0
_________________________________________________________________
```

### 3 | Training the model

- Fit model on <code>X_train</code> & validate on <code>X_val</code> w/ the corresponding <code>labels</code>

```python
history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=3, 
                    validation_data=(X_val, y_val))
```

```
Epoch 1/3
782/782 [==============================] - 16s 18ms/step - loss: 0.3986 - accuracy: 0.8170 - val_loss: 0.3362 - val_accuracy: 0.8658
Epoch 2/3
782/782 [==============================] - 14s 18ms/step - loss: 0.2114 - accuracy: 0.9187 - val_loss: 0.3773 - val_accuracy: 0.8482
Epoch 3/3
782/782 [==============================] - 14s 19ms/step - loss: 0.1372 - accuracy: 0.9519 - val_loss: 0.3639 - val_accuracy: 0.8569
```

### 4 | Some other options

- Some other options we can try, <code>LSTM</code> layer variation (add extra layer), which does improve the model

```
# Using 2 LSTM layer
Epoch 1/3
782/782 [==============================] - 28s 31ms/step - loss: 0.4043 - accuracy: 0.8163 - val_loss: 0.3337 - val_accuracy: 0.8594
Epoch 2/3
782/782 [==============================] - 23s 29ms/step - loss: 0.2118 - accuracy: 0.9195 - val_loss: 0.3532 - val_accuracy: 0.8706
Epoch 3/3
782/782 [==============================] - 23s 29ms/step - loss: 0.1405 - accuracy: 0.9483 - val_loss: 0.4333 - val_accuracy: 0.8617
```
  
- A smaller number of dictionary words during <code>tokenisation</code>, which clearly shows that a smaller number of tokenisation words reduces the <code>validation</code> accuracy
  
  
```
# Using 1000 word limit dictionary
Epoch 1/3
782/782 [==============================] - 17s 20ms/step - loss: 0.4821 - accuracy: 0.7664 - val_loss: 0.3737 - val_accuracy: 0.8361
Epoch 2/3
782/782 [==============================] - 14s 18ms/step - loss: 0.3991 - accuracy: 0.8272 - val_loss: 0.5943 - val_accuracy: 0.6830
Epoch 3/3
782/782 [==============================] - 14s 18ms/step - loss: 0.4171 - accuracy: 0.8083 - val_loss: 0.3665 - val_accuracy: 0.8458
```

- We could have also changed the architecture of the model & used <code>BiLSTM</code> over <code>LSTM</code>

```python
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
```
