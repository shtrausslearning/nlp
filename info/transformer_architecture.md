
### Transformer Architecture

The original transformer is based on the `encoder-decoder` architecture, used for translation

The archictecture has two components:
- `Encoder` (Converts a seq of tokens into a seq of embedding vectors - hidden state)
- `Decoder` (Uses the `encoder` hidden state to iteratively generate seq of tokens, one by one)

Some things that categorise `Transformers`:
- Input seq is `tokenised` & converted into `embeddings`
- The `attention` mechanism is not aware of the relative positions of `tokens`
- `Token embeddings` & combined with `positional embeddings` (contain positional about each `token`)
- `Encoder` (stack of encoder layers/blocks) - same as `convolutional` layer stacking
- `Decoder` (stack of decoder layer/blocks)
- `Encoder` output passed to each `decoder` layer, `decoder` then generates a prediction for most probable `token` to come next in the sequence
	- Output of this step, fed back into the `decoder` to generate the next token 
	- Process is repeated until it predicts **[EOS]** token

Example: 
- English to German Sequence to Sequence problem
- Decoder has alredy predicted `Die` `Zeit`, predicting next token 
 - next step: 	`Die` `Zeit` + `encoder` outputs -> `fliegt`
 - next step: `Di e` `Zeit` `fliegt` + `encoder` outputs -> `[EOS]` 
 - Repeated until `[EOS]` or until max length reched