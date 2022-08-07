
## Text Classification 

- IMDB sentiment classification dataset (unprocessed variant) (Text sentiment classification)
- Common <code>Binary</code> Classification problem with text based preprocessing required

### 1 | Get Dataset

- We'll need to obtain the dataset from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
- Which we can do using the <code>!curl</code> command

```
Other usable commands in jupyter:
- !curl 
- !tar
- !ls 
- !rm 
- !curl
```

```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

```python
import tensorflow as tf
import numpy as np

!ls aclImdb
!ls aclImdb/test
!ls aclImdb/train
# !rm -r aclImdb/train/unsup # remove folder
```

```
README     imdb.vocab imdbEr.txt test       train
labeledBow.feat neg             pos             urls_neg.txt    urls_pos.txt
labeledBow.feat pos             urls_neg.txt    urls_unsup.txt
neg             unsupBow.feat   urls_pos.txt
```

### 2 | Visualise some samples from the dataset

- We can visualise the fil in text format using the <code>!cat</code> command
- We can note that the data contains <code>HTML</code> tag <code>br /</code>, so our data will need to be processed

```python
# read file
!cat aclImdb/train/pos/6248_7.txt
```

```
Being an Austrian myself this has been a straight knock in my face. Fortunately 
I don't live nowhere near the place where this movie takes place but unfortunately 
it portrays everything that the rest of Austria hates about Viennese people 
(or people close to that region). And it is very easy to read that this is exactly 
the directors intention: to let your head sink into your hands and say "Oh my god, 
how can THAT be possible!". No, not with me, the (in my opinion) totally exaggerated 
uncensored swinger club scene is not necessary, I watch porn, sure, but in this context 
I was rather disgusted than put in the right context.<br /><br />This movie tells a 
story about how misled people who suffer from lack of education or bad company try to 
survive and live in a world of redundancy and boring horizons. A girl who is treated 
like a whore by her super-jealous boyfriend (and still keeps coming back), a female teacher 
who discovers her masochism by putting the life of her super-cruel "lover" on the line, an 
old couple who has an almost mathematical daily cycle (she is the "official replacement" of 
his ex wife), a couple that has just divorced and has the ex husband suffer under the acts 
of his former wife obviously having a relationship with her masseuse and finally a crazy 
hitchhiker who asks her drivers the most unusual questions and stretches their nerves by 
just being super-annoying.<br /><br />After having seen it you feel almost nothing. 
You're not even shocked, sad, depressed or feel like doing anything... Maybe that's why 
I gave it 7 points, it made me react in a way I never reacted before. 
If that's good or bad is up to you!
```

### 3 | Create a dataset

- Our dataset is loaded loaded in folder <code>aclImdb</code>, which contains two folders <code>train</code> & <code>test</code>
- Each folder contains <code>txt</code> files, containing a single review, <code>label</code> will be separate 
- We can utilise the <code>text_dataset_from_directory</code> function to load the data into a dataset (<code>BatchDataset</code>)

```python
from tensorflow.keras.preprocessing import text_dataset_from_directory

# Training set (80%)
raw_train_ds = text_dataset_from_directory("aclImdb/train",
                                           batch_size=32,
                                           validation_split=0.2,
                                           subset="training",
                                           seed=1337)

# Validation set (20%)
raw_val_ds = text_dataset_from_directory("aclImdb/train",
                                        batch_size=32,
                                        validation_split=0.2,
                                        subset="validation",
                                        seed=1337)

# Test set
raw_test_ds = text_dataset_from_directory("aclImdb/test",
                                          batch_size=32)

# print(f"Batches in raw_train_ds: {raw_train_ds.cardinality()}")
# print(f"Batches in raw_val_ds: {raw_val_ds.cardinality()}")
# print(f"Batches in raw_test_ds: {raw_test_ds.cardinality()}")

```

```
Found 25000 files belonging to 2 classes.
Using 20000 files for training.
Found 25000 files belonging to 2 classes.
Using 5000 files for validation.
Found 25000 files belonging to 2 classes.
```

- Having loaded the data into the <code>BatchDataset</code>, which are in <code>tensor</code> format
- We can visualise the data that has been read using <code>.take()</code>, which contains the **text batches** & corresponding **labels** 
- <code>labels</code> are created for each folder

```python

# Evaluate tensor w/ .numpy()
for text_batch, label_batch in raw_train_ds.take(1):
    print(text_batch.numpy()[31])
    print(label_batch.numpy()[31])
```

```
b'A man brings his new wife to his home where his former wife died of an "accident". His new wife has just been released from an institution and is also VERY rich! All of the sudden she starts hearing noises and seeing skulls all over the place. Is she going crazy again or is the first wife coming back from the dead? <br /><br />You\'ve probably guessed the ending so I won\'t spell it out. I saw this many times on Saturday afternoon TV as a kid. Back then, I liked it but I WAS young. Seeing it now I realize how bad it is. It\'s horribly acted, badly written, very dull (even at an hour) and has a huge cast of FIVE people (one being the director)! Still it does have some good things about it. <br /><br />The music is kinda creepy and the setting itself with the huge empty house and pond nearby is nicely atmospheric. There also are a few scary moments (I jumped a little when she saw the first skull) and a somewhat effective ending. All in all it\'s definitely NOT a good movie...but not a total disaster either. It does have a small cult following. I give it a 2.<br /><br />Also try to avoid the Elite DVD Drive-in edition of it (it\'s paired with "Attack of the Giant Leeches"). It\'s in TERRIBLE shape with jumps and scratches all over. It didn\'t even look this bad on TV!'
0
```

### 4 | Data Preparation

- Our data contains HTML break tags of the form '<br />', and is in raw format
- The default standardiser in <code>TextVectorization</code> (lower_and_strip_punctuation) will not remove them
- We need to create a custom function <code>custom_standardisation</code> that will **replace** them with **" "**
- Each review will be limited to <code>sequence_length = 500</code> 

```python

from tensorflow.keras.layers import TextVectorization
import string, re
 
# [1] Create Custom Standardisation Function removes html breaks
  
def custom_standardisation(input_data):
    lowercase = tf.strings.lower(input_data) # lowercase
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ") # regex replace
    return tf.strings.regex_replace(stripped_html, 
                                    f"[{re.escape(string.punctuation)}]", "")

# [2] Model constants
  
max_features = 20000
embedding_dim = 128
sequence_length = 500
 
#  [3] Instantiate our text vectorization layer

# We are using this layer to normalize, split, and map strings to integers, so we set our 'output_mode' to 'int'

# Note that we're using:
# - the default split function, 
# - custom standardization defined above.
# - set an explicit maximum sequence length, since the CNNs later in our model won't support ragged sequences

vectorize_layer = TextVectorization(standardize=custom_standardisation,
                                    max_tokens=max_features,
                                    output_mode="int",
                                    output_sequence_length=sequence_length)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary
# You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

```

### 5 | Vectorising the data

There are a couple of approaches we can take in utilisation of <code>TextVectorization</code>
- [1] Modify the dataset directly before feeding the data into our model, creating <code>train_ds</code> & <code>val_ds</code>
- [2] Make the layer part of the model, using <code>raw_train_ds</code> & <code>raw_val_ds</code> as inputs

#### 5.1 | APPROACH 1

```python

# Function that modifies the text input (used with map)
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data 
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

```

```python
for i,j in train_ds.take(1):
    print(i,j)
```

```
tf.Tensor(
[[   10    19  1692 ...     0     0     0]
 [    1  1017    19 ...     0     0     0]
 [   11    38  2925 ...     0     0     0]
 ...
 [   11   203    76 ...     0     0     0]
 [   10     7     2 ...     0     0     0]
 [    2 11069     7 ...     0     0     0]], shape=(32, 500), dtype=int64) tf.Tensor([1 1 0 1 0 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 0 0 0], shape=(32,), dtype=int32)
```

- Define the <code>model</code> architecture

```python

# Create Binary Classifier Model

from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
              
```

- Train the model on the updated dataset <code>train_ds</code> & <code>val_ds</code>

```python
# Fit the model using the raw train and test datasets
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=3)
```

```
Epoch 1/3
625/625 [==============================] - 76s 121ms/step - loss: 0.4820 - accuracy: 0.7337 - val_loss: 0.3080 - val_accuracy: 0.8714
Epoch 2/3
625/625 [==============================] - 66s 106ms/step - loss: 0.2194 - accuracy: 0.9123 - val_loss: 0.3396 - val_accuracy: 0.8622
Epoch 3/3
625/625 [==============================] - 64s 102ms/step - loss: 0.1020 - accuracy: 0.9630 - val_loss: 0.4204 - val_accuracy: 0.8744
```
- Evaluate the model on the adjusted dataset <code>train_ds</code>

```python
model.evaluate(test_ds)
```

```
782/782 [==============================] - 19s 24ms/step - loss: 0.4693 - accuracy: 0.8566
[0.469294935464859, 0.8565599918365479]
```

#### 5.2 | APPROACH 2

- We will make the <code>vectorize_layer</code> part of the model architecture (similar to <code>TextVectorization.py</code>)
- Compile the model using <code>loss</code> **binary_crossentropy** & <code>optimiser</code>adam</code> using default settings
- We will monitor only the <code>accuracy</code> metric for both <code>train</code> & <code>validation</code> datasets
  
```python
# Create Binary Classifier Model

from tensorflow.keras import layers

# A integer input for vocab indices.
text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = layers.Embedding(max_features + 1, embedding_dim)(x)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(text_input, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
```

- Train the model on the raw dataset <code>raw_train_ds</code>, <code>raw_val_ds</code>
  
```python
# Fit the model using the raw train and test datasets
history = model.fit(raw_train_ds, 
                    validation_data=raw_val_ds,
                    epochs=3)
```
  
  
```
Epoch 1/3
625/625 [==============================] - 76s 121ms/step - loss: 0.4820 - accuracy: 0.7337 - val_loss: 0.3080 - val_accuracy: 0.8714
Epoch 2/3
625/625 [==============================] - 66s 106ms/step - loss: 0.2194 - accuracy: 0.9123 - val_loss: 0.3396 - val_accuracy: 0.8622
Epoch 3/3
625/625 [==============================] - 64s 102ms/step - loss: 0.1020 - accuracy: 0.9630 - val_loss: 0.4204 - val_accuracy: 0.8744
```

- Evaluating the model on <code>raw_test_ds</code>

```python
inputs = tf.keras.Input(shape=(1,), dtype="string") # A string input
indices = vectorize_layer(inputs) # Turn strings into vocab indices
outputs = model(indices) # Turn vocab indices into predictions

# Our end to end model
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(loss="binary_crossentropy",
                         optimizer="adam", 
                        metrics=["accuracy"])

# Test it with `raw_test_ds`, which yields raw strings
end_to_end_model.evaluate(raw_test_ds)
```

```
782/782 [==============================] - 27s 34ms/step - loss: 1.0694 - accuracy: 0.8029
[1.0694401264190674, 0.8028799891471863]
```
