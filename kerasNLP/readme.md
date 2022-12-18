
### Keras NLP

- <code>KerasNLP</code> is a toolbox of modular building blocks (layers, metrics, etc.) that NLP engineers can leverage to develop production-grade, state-of-the-art training and inference pipelines for common NLP workflows.
- In this section, we'll look at examples of the class implementation in NPL workflows

```python
!pip install keras-nlp --upgrade
```

### Tokenisers

For <code>tokenisation</code>, we have the following options, calling from (<code>keras_nlp.tokenizers</code>)

- [<code>Tokenizer</code>](https://keras.io/api/keras_nlp/tokenizers/tokenizer/) base class
- [<code>WordPieceTokenizer</code>](https://keras.io/api/keras_nlp/tokenizers/word_piece_tokenizer/) ([example](https://github.com/shtrausslearning/nlp/blob/main/kerasNLP/wordpiecetokeniser_example.ipynb))
- SentencePieceTokenizer
- ByteTokenizer
- UnicodeCharacterTokenizer

### Layers

When creating a transformer model, we have the following <code>layers</code> available to us from (<code>keras_nlp.layers</code>)

- [<code>TransformerEncoder</code>](https://keras.io/api/keras_nlp/layers/transformer_encoder/)
- [<code>TransformerDecoder</code>](https://keras.io/api/keras_nlp/layers/transformer_decoder/)
- FNetEncoder
- PositionEmbedding
- SinePositionEncoding
- TokenAndPositionEmbedding
- MLMMaskGenerator
- MLMHead
- [<code>StartEndPacker</code>](https://keras.io/api/keras_nlp/layers/start_end_packer/)
- MultiSegmentPacker

### Metrics

- Perplexity metric
- RougeL metric
- RougeN metric

### Utils

- greedy_search function
- top_k_search function
- top_p_search function
- random_search function
- beam_search function
