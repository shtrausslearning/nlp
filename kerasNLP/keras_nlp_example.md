### Keras NLP

- We utilise <code>WordPieceTokenizer</code> to **tokenise**, which is a `subword` tokenisation method

```python
sentences = ["The quick brown fox jumped.", 
             "The fox slept."]
```

- Using an existing vocabulary <code>vocab</code> & a <code>padding</code> option of 10 words 


```python

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import TransformerEncoder,TokenAndPositionEmbedding
from keras_nlp.tokenizers import WordPieceTokenizer

# Tokenize some inputs with a binary label
vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]

# Wordpiece Tokeniser
tokenizer = WordPieceTokenizer(vocabulary=vocab,   # vocabulary
                               sequence_length=10) # padding

# Tokenise input sentences
x, y = tokenizer(sentences), 
                 tf.constant([1, 0])
print(x,y)
```

```
tf.Tensor(
[[1 2 3 4 5 6 0 7 0 0]
 [1 6 0 7 0 0 0 0 0 0]], shape=(2, 10), dtype=int32) tf.Tensor([1 0], shape=(2,), dtype=int32)
```

- Create a model transformer 

```python
# Create a transformer
inputs = keras.Input(shape=(None,), dtype="int32")

# A layer which sums a token and position embedding
x = TokenAndPositionEmbedding(vocabulary_size=len(vocab),
                              sequence_length=10,
                              embedding_dim=16)(inputs)

# Encoder
x = TransformerEncoder(num_heads=4,
                       intermediate_dim=32)(x)

# GlobalAveragePooling1D & Dense layer from keras.layers
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 16)         288       
 g_2 (TokenAndPositionEmbedd                                     
 ing)                                                            
                                                                 
 transformer_encoder_2 (Tran  (None, None, 16)         2224      
 sformerEncoder)                                                 
                                                                 
 global_average_pooling1d_2   (None, 16)               0         
 (GlobalAveragePooling1D)                                        
                                                                 
 dense_2 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 2,529
Trainable params: 2,529
Non-trainable params: 0
_________________________________________________________________
```

```python
# Regression Metrics in Tensorflow/Keras
m = tf.keras.metrics.RootMeanSquaredError()
# m = tf.keras.metrics.MeanSquaredError()
# m = tf.keras.metrics.MeanAbsoluteError()
# m = tf.keras.metrics.MeanAbsolutePercentageError()
# m = tf.keras.metrics.MeanSquaredLogarithmicError()
# m = tf.keras.metrics.LogCoshError()
# m = tf.keras.metrics.CosineSimilarity(axis=1)

# Run a single batch of gradient descent.
model.compile(optimizer="rmsprop", 
              loss="binary_crossentropy",
              metrics=[m],
              jit_compile=True)

# model.train_on_batch(x,y)
history = model.fit(x,y,epochs=4)
```

```
Epoch 1/4
1/1 [==============================] - 5s 5s/step - loss: 0.6971 - mean_squared_error: 0.2520
Epoch 2/4
1/1 [==============================] - 0s 8ms/step - loss: 0.6369 - mean_squared_error: 0.2221
Epoch 3/4
1/1 [==============================] - 0s 21ms/step - loss: 0.5962 - mean_squared_error: 0.2019
Epoch 4/4
1/1 [==============================] - 0s 29ms/step - loss: 0.5641 - mean_squared_error: 0.1862
```
