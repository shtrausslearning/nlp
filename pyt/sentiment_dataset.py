from torch.utils.data import Dataset

def SentimentDataset(Dataset):

    def __init__(self,data_path,tokeniser):
        self.tokeniser = tokeniser

        # read text data

        self.data = []
        with open(data_path,'r') as f:
            for line in f:
                text, label = line.strip().split('\t')
                self.data.append((text,int(label)))

    def __len__(self):
        return len(self.data)

    def __get__(self,index):
        text,label = self.data[index] # get document
        tokens = self.tokeniser.tokenise(text)
        encoded = self.tokeniser.convert_tokens_to_ids(tokens)
        return encoded, label





