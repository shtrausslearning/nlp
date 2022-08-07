Keras' <code>TextVectorisation</code> layer usage

```python
text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])

max_features = 5000  # Maximum vocab size
max_len = 5          # Sequence length to pad the outputs to
embedding_dims = 2

# Create layer
vectorize_layer = TextVectorization(standardize='lower_and_strip_punctuation', # default
                                    max_tokens=max_features,    
                                    output_mode='int', 
                                    split='whitespace',   # default
                                    output_sequence_length=max_len)  
                      
# call `adapt` on the text-only dataset to create the vocabulary              
vectorize_layer.adapt(text_dataset.batch(32))

# Create model that uses the layer
model = tf.keras.models.Sequential() 

# Start by creating an explicit input layer. 
# Needs to have a shape of (1,) & dtype 'string'. 
# Need to guarantee that there is exactly one string input per batch)
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))  

# The first layer in our model is the vectorization layer
# After this layer, we have a tensor of shape (batch_size, max_len) 
# containing vocab indices.  
model.add(vectorize_layer)

# list of two lists
input_data = [["foo qux bar"], 
              ["qux baz"]] 

model.predict(input_data)
```

```
array([[2, 1, 4, 0, 0],
       [1, 3, 0, 0, 0]])
```
