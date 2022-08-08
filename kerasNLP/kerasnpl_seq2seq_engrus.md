## seq2seq task 


In this notebook, we'll look at the following problem:

- English-to-Russian translation with <code>KerasNLP</code> module
- We'll be creating a <code>encoder</code> | <code>decoder</code> transformer model

KerasNLP

- <code>KerasNLP</code> is a toolbox of modular building blocks (layers, metrics, etc.) that NLP engineers can leverage to develop production-grade, state-of-the-art training and inference pipelines for common NLP workflows
- It is a separate module, that works with Keras/Tensorflow & aims to simplify **NLP** workflows

```python
!pip install keras-nlp --upgrade
```

### 1 | Create list of text pairs

We'll be using <code>eng-rus.txt</code>, which is available from the **[following source](http://www.manythings.org/anki/)**
Let's take a peek at the contents of the file:

```
Go.	Марш!	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #1159202 (shanghainese)
Go.	Иди.	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #5898247 (marafon)
Go.	Идите.	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #5898250 (marafon)
```

We need to create a **training pair**, which should look as follows, an english & subsequently russian translation pair

```
('go.', 'марш!')
```

We need to apply some extra **string** operations, <code>get_pairs</code> can be used to create a list of tuple pairs

```python
def get_pairs():
    
    text_file = pathlib.Path('/kaggle/input/seq2seq-task/eng-rus.txt')

    text_pairs = []
    with open(text_file) as file:
        for ii,line in enumerate(file):
            tline = line.split('CC-BY')[0]
            tline = tline.rstrip()
            eng, fl = tline.split('\t')
            eng = eng.lower(); fl = fl.lower()
            text_pairs.append((eng,fl))
            
    print('Created pairs!')
    return text_pairs

text_pairs = get_pairs()
```

```
Created pairs!
```

### 2 | Divide dataset into subsets

Having created training pairs (text_pairs), we need to split the entire dataset into <code>training</code>, <code>test</code> & <code>validation</code> sets

```python
# Split the pairs into training/validation/test sets
def split_data(text_pairs,val_n=0.15):

    random.shuffle(text_pairs) # shuffle list 
    num_val_samples = int(val_n * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples

    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]
    return train_pairs,val_pairs,test_pairs

train_pairs,val_pairs,test_pairs = split_data(text_pairs,0.15)

print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

```
311211 training pairs
66688 validation pairs
66688 test pairs
```

### 3 | Tokenise Text Pairs

- Using a **[subword tokenizer](https://www.tensorflow.org/text/guide/subwords_tokenizer?hl=en)**, we can create a **dictionary** for both languages from both corpuses 
- The main advantage of a **subword tokenizer**
   - It interpolates between **work based** and **character-based** tokenization 
   - Common words get a slot in the vocabulary, but the tokenizer can fall back to word pieces and individual characters for unknown words

Creating tokenised data:

- <code>train_word_piece</code> creates a **vocabulary** using <code>bert_vocab_from_dataset</code>
- <code>WordPieceTokenizer</code> takes in the **vacabulary** dictionary & tokenises the two language pair arrays

```python

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

def tokenise(train_pairs):

    def train_word_piece(text_samples, vocab_size, reserved_tokens):
        
        word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
        
        bert_vocab_args = {
            'vocab_size':vocab_size,                     # The target vocabulary size
            'reserved_tokens':reserved_tokens,
            'bert_tokenizer_params':{"lower_case": True}} # Arguments for `text.BertTokenizer
        
        vocab = bert_vocab.bert_vocab_from_dataset(
            word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args)
        return vocab

    # Build a vocabulary for WordPieceTokeniser | Train Wordpiece on a corpus

    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

    eng_samples = [text_pair[0] for text_pair in train_pairs]
    fl_samples = [text_pair[1] for text_pair in train_pairs]
    
    eng_vocab = train_word_piece(eng_samples, 
                                 cfg.ENG_VOCAB_SIZE,
                                 reserved_tokens)

    fl_vocab = train_word_piece(fl_samples, 
                                cfg.FL_VOCAB_SIZE,
                                reserved_tokens)

    # Show vocabulary info

    print(f'Vocab length {len(eng_vocab)}')
    print(f'Vocab length {len(fl_vocab)}')

    print("English Tokens: ", eng_vocab[100:110])
    print("Russian Tokens: ", fl_vocab[100:110])

    # Tokeniser
    eng_tokeniser = WordPieceTokenizer(vocabulary=eng_vocab, 
                                       lowercase=False)
    fl_tokeniser = WordPieceTokenizer(vocabulary=fl_vocab,
                                      lowercase=False)

    return eng_tokeniser, fl_tokeniser

eng_tokeniser, fl_tokeniser = tokenise(train_pairs)
```

```
Vocab length 5779
Vocab length 12600
English Tokens:  ['about', 'here', 'there', 'has', 'going', 'tell', 'will', 'one', 'told', 'who']
Russian Tokens:  ['мы', 'меня', 'он', 'как', 'тому', 'все', 'бы', 'тебе', 'сказал', 'чтобы']
```

### 4 | Create dataset

```python
from keras_nlp.layers import StartEndPacker

def preprocess_batch(eng, fl):
    
    batch_size = tf.shape(fl)[0]

    eng = eng_tokeniser(eng)
    fl = fl_tokeniser(fl)

    # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
    eng_start_end_packer = StartEndPacker(sequence_length=cfg.MAX_SEQUENCE_LENGTH, # padding
                                          pad_value=eng_tokeniser.token_to_id("[PAD]"))
    eng = eng_start_end_packer(eng)

    # Add special tokens (`"[START]"` and `"[END]"`) to `spa` and pad it as well.
    fl_start_end_packer = StartEndPacker(sequence_length=cfg.MAX_SEQUENCE_LENGTH + 1,
                                         start_value=fl_tokeniser.token_to_id("[START]"),
                                         end_value=fl_tokeniser.token_to_id("[END]"),
                                         pad_value=fl_tokeniser.token_to_id("[PAD]"))
    fl = fl_start_end_packer(fl)

    return ({"encoder_inputs": eng,"decoder_inputs": fl[:, :-1]},
            fl[:, 1:])


def make_dataset(pairs):

    eng_texts, fl_texts = zip(*pairs) # unzip tuple pair
    eng_texts = list(eng_texts) # tuple to list
    fl_texts = list(fl_texts)  # tuple to list

    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fl_texts))
    dataset = dataset.batch(cfg.BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

print(f'training dataset: {len(train_ds)}')
print(f'validation dataset: {len(val_ds)}')
```

```
training dataset: 4863
validation dataset: 1042
```


### 5 | Create Transformer Model

```python
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder


# Encoder

# input to encoder model
encoder_inputs = keras.Input(shape=(None,), 
                             dtype="int64",
                             name="encoder_inputs")

x = TokenAndPositionEmbedding(vocabulary_size=cfg.ENG_VOCAB_SIZE,
                              sequence_length=cfg.MAX_SEQUENCE_LENGTH,
                              embedding_dim=cfg.EMBED_DIM,
                              mask_zero=True)(encoder_inputs)

# Create single transformer encoder layer
encoder_outputs = TransformerEncoder(intermediate_dim=cfg.INTERMEDIATE_DIM, 
                                     num_heads=cfg.NUM_HEADS)(inputs=x)
                                     
encoder = keras.Model(encoder_inputs, encoder_outputs)

# Decoder

encoded_seq_inputs = keras.Input(shape=(None,cfg.EMBED_DIM),
                                 name="decoder_state_inputs")

# input into decoder model
decoder_inputs = keras.Input(shape=(None,),
                             dtype="int64", 
                             name="decoder_inputs")

x = TokenAndPositionEmbedding(vocabulary_size=cfg.FL_VOCAB_SIZE,
                              sequence_length=cfg.MAX_SEQUENCE_LENGTH,
                              embedding_dim=cfg.EMBED_DIM,
                              mask_zero=True)(decoder_inputs)

x = TransformerDecoder(intermediate_dim=cfg.INTERMEDIATE_DIM,
                       num_heads=cfg.NUM_HEADS)(decoder_sequence=x,
                                                encoder_sequence=encoded_seq_inputs)
                       
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(cfg.FL_VOCAB_SIZE, activation="softmax")(x)

decoder = keras.Model([decoder_inputs,encoded_seq_inputs],decoder_outputs)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model([encoder_inputs, decoder_inputs],
                          decoder_outputs,
                          name="transformer")

transformer.summary()
```

```
Model: "transformer"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 encoder_inputs (InputLayer)    [(None, None)]       0           []                               
                                                                                                  
 token_and_position_embedding (  (None, None, 256)   3850240     ['encoder_inputs[0][0]']         
 TokenAndPositionEmbedding)                                                                       
                                                                                                  
 decoder_inputs (InputLayer)    [(None, None)]       0           []                               
                                                                                                  
 transformer_encoder (Transform  (None, None, 256)   1315072     ['token_and_position_embedding[0]
 erEncoder)                                                      [0]']                            
                                                                                                  
 model_1 (Functional)           (None, None, 15000)  10203288    ['decoder_inputs[0][0]',         
                                                                  'transformer_encoder[0][0]']    
                                                                                                  
==================================================================================================
Total params: 15,368,600
Trainable params: 15,368,600
Non-trainable params: 0
__________________________________________________________________________________________________
```


