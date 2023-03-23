
# Example of Tast text model 

import torch.nn as nn 
import torch.nn.functional as f

class FastText(nn.Module):
    
    def __init__(self,vocab_size,embed_dim,num_class):
        
        super(FastTextself).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.fc = nn.Linear(embed_dim,num_class)
        self.softmax = nn.Softmax(dim=1)
        
    def __forward__(self,x):
        x = self.embedding(x) # convert input tokens to word vectors
        x = x.mean(axis=1) # average pooling layer
        x = self.fc(x)      # linear layer
        x = self.softmax(x) # non linear activation function
        return x
        

class FastText(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text) 
        embedded = embedded.permute(1,0,2) 
        pooled = f.avg_pool2d(embedded,(embedded.shape[1], 1)).squeeze(1)     
        return self.fc(pooled) 
            
      
    
