import pandas as pd
import numpy as np
import nltk
import re
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm.notebook import tqdm

def make_context_vector(context, word2id):
    idxs = [word2id[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

# Load a single document

lst_data = ['With Leonard, Howard, Raj, and Amy accomplishing so much on their respective projects, Sheldon is forced to admit he has nothing important upon which to work.', 
            'He makes Amy leave the apartment for a few days so he can focus, but cannot come up with any ideas and calls his mother as a distraction.',
            'Leonard and Amy have fun recreating experiments from when they were growing up, boring Penny, so she eats with Sheldon as he mulls over his scientific studies.'
            'Penny helps him realize that his study of dark matter is his rebound science from string theory, which Sheldon admits he never truly disregarded, but explaining string theory to her inspires Sheldon, helping him discover a potential breakthrough in the field.',
            'Meanwhile, Howard is too busy with his family to be in the band with Raj, so Raj brings in Bert.',
            'But when Howard annoys Bernadette by writing an astronaut-themed musical while she is on bed rest, she makes him rejoin the band.',
            "The three are poorly received at a Bar mitzvah after singing Bert's original song about the boulder from Raiders of the Lost Ark."]

corpus = pd.DataFrame(lst_data,columns=['text'])
corpus['length'] = corpus['text'].str.len()
ldf = corpus.sort_values(by='length',ascending=False)
raw_text = ldf.iloc[[0]]['text'].str.split().tolist()[0]

print('Raw Documents:')
print(raw_text)
print(len(raw_text),'words')

# ['Leonard', 'and', 'Amy', 'have', 'fun', 'recreating', 'experiments', 'from', 'when', 'they', 'were', 'growing', 'up,', 'boring', 'Penny,', 'so', 'she', 'eats', 'with', 'Sheldon', 'as', 'he', 'mulls', 'over', 'his', 'scientific', 'studies.Penny', 'helps', 'him', 'realize', 'that', 'his', 'study', 'of', 'dark', 'matter', 'is', 'his', 'rebound', 'science', 'from', 'string', 'theory,', 'which', 'Sheldon', 'admits', 'he', 'never', 'truly', 'disregarded,', 'but', 'explaining', 'string', 'theory', 'to', 'her', 'inspires', 'Sheldon,', 'helping', 'him', 'discover', 'a', 'potential', 'breakthrough', 'in', 'the', 'field.']
# 67 words

'''

Prepare Data for CBOW model

'''

class prepare:
    
    def __init__(self):
        
        self.vocab = set(raw_text)          
        self.vocab_size = len(self.vocab)     # size of dictionary vocabulary
        self.word2id = {word:ix for ix, word in enumerate(self.vocab)} # dictionary for conversion 
        self.id2word = {ix:word for ix, word in enumerate(self.vocab)} # dictionary for conversion
        self.context_size = 2       # document contect range (+/-)
        self.embedding_size = 20   # embedding size
        self.epochs = 40           # number of iterations
        self.lr = 0.001             # learning rate
        
        print('\nInput Data:')
        print('--------------------------------')
        print(f'Vocabulary Size: {self.vocab_size}')
        print(f'Context Size: {self.context_size}')
        print(f'Embedding Vector Size: {self.embedding_size}')
        print(f'Model Training Epochs: {self.epochs}')
        print(f'Model Learning Rate: {self.lr}\n')
        
        # Prepare Context array & target vector
        data = []
        for i in range(2, len(raw_text) - 2):
            context = [raw_text[i - 2], raw_text[i - 1],
                       raw_text[i + 1], raw_text[i + 2]]
            target = raw_text[i]
            data.append((context, target))
            
        self.data = data
        
cfg = prepare()

# Input Data:
# --------------------------------
# Vocabulary Size: 60
# Context Size: 2
# Embedding Vector Size: 20
# Model Training Epochs: 40
# Model Learning Rate: 0.001

''' 

CBOW Model Architecture 

'''

class CBOW(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        
        super(CBOW, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.active1 = nn.ReLU()
        
        # linear layer
        self.linear2 = nn.Linear(128, vocab_size)
        self.active2 = nn.LogSoftmax(dim = -1)
        
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds); out = self.active1(out)
        out = self.linear2(out); out = self.active2(out)
        return out

    def get_word_embedding(self, word):
        word = torch.tensor([cfg.word2id[word]])
        return self.embeddings(word).view(1,-1)

model = CBOW(cfg.vocab_size,cfg.embedding_size)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

'''

Train CBOW Model

'''

lst_loss = []
for epoch in tqdm(range(cfg.epochs)):
    
    total_loss = 0.0
    for context, target in cfg.data:
        
        context_vector = make_context_vector(context, cfg.word2id)  
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([cfg.word2id[target]]))

    #optimize at the end of each epoch
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    lst_loss.append(float(total_loss.detach().numpy()))
    
pd.options.plotting.backend = "plotly"
df_loss = pd.DataFrame({'loss':lst_loss})
print(df_loss.iloc[[-1]])

#    loss 
#39  127.657028

'''

Extracting Embedding Data

'''

print('\nEmbedding Vector for "Leonard:"')
model.get_word_embedding('Leonard').detach().numpy()

# array([[-0.48914912, -1.3664836 , -1.2825114 , -0.26156828, -0.88690007,
#         -0.1482072 ,  0.26872754,  1.2552432 ,  1.1077945 ,  0.7150579 ,
#          0.9777638 ,  0.19413915,  1.757421  ,  0.6553982 , -1.0077755 ,
#          0.2781264 ,  1.779956  ,  1.5397804 ,  0.8350439 ,  1.4066876 ]],
#       dtype=float32)
