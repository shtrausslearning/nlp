from torch.utils.data import Dataset

class lmDataset(Dataset):

    def __init__(self,path,tokeniser,seq_len):
        self.tokeniser = tokeniser
        self.seq_len = seq_len

        self.data = []
        with open(path,'r') as f:
            for line in f:
                text = line.strip()
                if(len(text) > self.seq_len):

                    # split text into parts of length seq_len
                    for i in range(0,len(text) - self.seq_len + 1,self.seq_len):
                        part = text[i:i+self.seq_len]
                        self.data.append(self.tokeniser.encode(part))
                
                else:
                    self.data.append(self.tokeniser.encode(text))


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        # get sequence of len seq_len starting from index
        seq = self.data[index][:self.seq_len] 
        
        # pad sequence if it's shorter than seq_len
        padding = [self.tokeniser.pad_token_id] * (self.seq_len - len(seq))
        seq += padding

        input_seq = seq[:-1] # input sequence
        target_seq = seq[1:] # target sequence
        return input_seq,target_seq 




