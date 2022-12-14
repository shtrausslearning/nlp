
from transformers import AutoTokenizer

# Input Text
text = 'Tokenising text is a core task of NLP.'

# AutoTokeniser Subword Tokenisation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoded_text = tokenizer(text)
print(encoded_text['input_ids'])
print(encoded_text['attention_mask'],'\n')

print('')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Tokeniser model input names: {tokenizer.model_input_names}')

# generate token dictionary
print('token dictionary')
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
