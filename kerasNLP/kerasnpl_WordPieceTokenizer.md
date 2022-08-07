```python
from keras_nlp.tokenizers import WordPieceTokenizer

vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
inputs = ["The quick brown fox."]
tokenizer = WordPieceTokenizer(vocabulary=vocab,sequence_length=10)

print(inputs)
print(tokenizer(inputs))  # short for .tokenise
print(tokenizer.tokenize(inputs)) 
print(tokenizer.get_vocabulary())
print(f'Size of vocabulary: {tokenizer.vocabulary_size()}')
print(tokenizer.detokenize([1,2,3,4,5,6,7,0,0,0]))
```

```
['The quick brown fox.']
tf.Tensor([[1 2 3 4 5 6 7 0 0 0]], shape=(1, 10), dtype=int32)
tf.Tensor([[1 2 3 4 5 6 7 0 0 0]], shape=(1, 10), dtype=int32)
ListWrapper(['[UNK]', 'the', 'qu', '##ick', 'br', '##own', 'fox', '.'])
Size of vocabulary: 8
tf.Tensor(b'the quick brown fox . [UNK] [UNK] [UNK]', shape=(), dtype=string)
```
