# List
corpus = ["After Sheldon and Leonard spend two months repairing Sheldon's DNA molecule model, everyone prepares to fly to Sweden for the Nobel Prize award ceremony.",
            "Howard and Bernadette nervously leave their kids for the first time with Stuart and Denise, while Raj leaves his dog with Bert."]

from transformers import AutoTokenizer

''' Working w/ Tokeniser (list) '''
print('Corpus documents:')
for document in corpus:
    print(document)

# AutoTokeniser Subword Tokenisation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoded_corpus = tokenizer(corpus,padding=True,truncation=True)

print(f'\ntokenised document:\n',encoded_corpus['input_ids'])            # encoded values
print(f'\nattention mask document:\n',encoded_corpus['attention_mask'],'\n')  # encoded values mask (for multiple documents)

# tokeniser information (general)
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Tokeniser model input names: {tokenizer.model_input_names}')

print('\nConvert IDs to Tokens')

lst_corpus = []
for document in encoded_corpus['input_ids']:
    tokens = tokenizer.convert_ids_to_tokens(document)
    lst_corpus.append(tokens)
    
print(lst_corpus)

print('\nConvert tokens to string')
for document in lst_corpus:
    print(tokenizer.convert_tokens_to_string(document))
    
# Corpus documents:
# After Sheldon and Leonard spend two months repairing Sheldon's DNA molecule model, everyone prepares to fly to Sweden for the Nobel Prize award ceremony.
# Howard and Bernadette nervously leave their kids for the first time with Stuart and Denise, while Raj leaves his dog with Bert.

# tokenised document:
#  [[101, 2044, 19369, 1998, 7723, 5247, 2048, 2706, 26296, 19369, 1005, 1055, 6064, 13922, 2944, 1010, 3071, 20776, 2000, 4875, 2000, 4701, 2005, 1996, 10501, 3396, 2400, 5103, 1012, 102], [101, 4922, 1998, 16595, 9648, 4674, 12531, 2681, 2037, 4268, 2005, 1996, 2034, 2051, 2007, 6990, 1998, 15339, 1010, 2096, 11948, 3727, 2010, 3899, 2007, 14324, 1012, 102, 0, 0]]

# attention mask document:
#  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] 

# Vocab size: 30522
# Max length: 512
# Tokeniser model input names: ['input_ids', 'attention_mask']

# Convert IDs to Tokens
# [['[CLS]', 'after', 'sheldon', 'and', 'leonard', 'spend', 'two', 'months', 'repairing', 'sheldon', "'", 's', 'dna', 'molecule', 'model', ',', 'everyone', 'prepares', 'to', 'fly', 'to', 'sweden', 'for', 'the', 'nobel', 'prize', 'award', 'ceremony', '.', '[SEP]'], ['[CLS]', 'howard', 'and', 'bern', '##ade', '##tte', 'nervously', 'leave', 'their', 'kids', 'for', 'the', 'first', 'time', 'with', 'stuart', 'and', 'denise', ',', 'while', 'raj', 'leaves', 'his', 'dog', 'with', 'bert', '.', '[SEP]', '[PAD]', '[PAD]']]

# Convert tokens to string
# [CLS] after sheldon and leonard spend two months repairing sheldon's dna molecule model, everyone prepares to fly to sweden for the nobel prize award ceremony. [SEP]
# [CLS] howard and bernadette nervously leave their kids for the first time with stuart and denise, while raj leaves his dog with bert. [SEP] [PAD] [PAD]
