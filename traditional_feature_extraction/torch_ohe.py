
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

import torch
import torch.nn.functional as F

inputs_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(inputs_ids,num_classes = len(token2id))
print(f' OHE size: {one_hot_encodings.shape}')

one_hot_encodings.numpy()  # to numpy array
one_hot_encodings.tolist() # to list

# [101, 19204, 9355, 3793, 2003, 1037, 4563, 4708, 1997, 17953, 2361, 1012, 102]
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 


# Vocab size: 30522
# Max length: 512
# Tokeniser model input names: ['input_ids', 'attention_mask']

# Convert IDs to Tokens
# ['[CLS]', 'token', '##ising', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '.', '[SEP]'] 

# Convert tokens to string
# [CLS] tokenising text is a core task of nlp. [SEP] 

# Token dictionary
# {'##ising': 0, '##p': 1, '.': 2, '[CLS]': 3, '[SEP]': 4, 'a': 5, 'core': 6, 'is': 7, 'nl': 8, 'of': 9, 'task': 10, 'text': 11, 'token': 12}

#  token2id 
#  [3, 12, 0, 11, 7, 5, 6, 10, 9, 8, 1, 2, 4] 

#  OHE size: torch.Size([13, 13])
# [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
