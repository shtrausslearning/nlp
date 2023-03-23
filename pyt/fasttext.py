
# Example of Tast text model 

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
        
        
        
            
      
    
