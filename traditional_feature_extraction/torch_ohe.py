
from transformers import AutoTokenizer

# Input Text
text = 'Tokenising text is a core task of NLP.'

''' Working w/ Tokeniser '''

# AutoTokeniser Subword Tokenisation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoded_text = tokenizer(text)
print(encoded_text['input_ids'])            # encoded values
print(encoded_text['attention_mask'],'\n')  # encoded values mask (for multiple documents)

print('')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Tokeniser model input names: {tokenizer.model_input_names}')

print('\nConvert IDs to Tokens')
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens,'\n')

print('Convert tokens to string')
print(tokenizer.convert_tokens_to_string(tokens),'\n')

''' Generate Vocabulary '''

# generate token dictionary
print('Token dictionary')
token2id = {ch: idx for idx,ch in enumerate(sorted(set(tokens)))}
print(token2id)

# generate ids for tokens via vocab dictionary
input_ids = [token2id[token] for token in tokens]

print('\n','token2id','\n',input_ids,'\n')

''' Generate Tokens (for df['text']) '''
# If we want to tokenise a column in a dataset

# Tokenisation function
def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

df_tokenised = tokenise(df["train"])

''' One Hot Encoding using PyTorch '''

import torch
import torch.nn.functional as F

inputs_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(inputs_ids,num_classes = len(token2id))
print(f' OHE size: {one_hot_encodings.shape}')

one_hot_encodings.numpy()  # to numpy array
one_hot_encodings.tolist() # to list
