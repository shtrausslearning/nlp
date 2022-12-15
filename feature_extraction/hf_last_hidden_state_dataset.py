# Pandas DataFrame
import pandas as pd
lst_data = ["After an unsuccessful visit to the high-IQ sperm bank, Dr. Leonard Hofstadter and Dr. Sheldon Cooper return home to find aspiring actress Penny is their new neighbor across the hall from their apartment. Sheldon thinks Leonard, who is immediately interested in her, is chasing a dream he will never catch. Leonard invites Penny to his and Sheldon's apartment for Indian food, where she asks to use their shower since hers is broken. While wrapped in a towel, she gets to meet their visiting friends Howard Wolowitz, a wannabe ladies' man who tries to hit on her, and Rajesh Koothrappali, who is unable to speak to her as he suffers from selective mutism in the presence of women. Leonard is so infatuated with Penny that, after helping her use their shower, he agrees to retrieve her TV from her ex-boyfriend Kurt. However, Kurt's physical superiority overwhelms Leonard's and Sheldon's combined IQ of 360, and they return without pants or TV. Penny, feeling bad, offers to take the guys out to dinner, initiating a friendship with them.",
            "After Sheldon and Leonard spend two months repairing Sheldon's DNA molecule model, everyone prepares to fly to Sweden for the Nobel Prize award ceremony. Howard and Bernadette nervously leave their kids for the first time with Stuart and Denise, while Raj leaves his dog with Bert. Penny has become pregnant, though she and Leonard are keeping it a secret. On the flight, Raj meets Sarah Michelle Gellar. Penny's frequent bathroom trips make Sheldon fear she is sick. Penny reveals her pregnancy to Sheldon but, instead of being excited for her, Sheldon is only selfishly relieved that he will not get sick, and he exposes the pregnancy, offending Leonard. At the hotel, a series of minor incidents with their kids make Howard and Bernadette want to go home. Much to their dismay, Sheldon is still insensitive. Amy furiously tells Sheldon he broke his friends' hearts and that people (sometimes including her) only tolerate him because he does not intentionally do so. Everyone decides to stay for the ceremony and Raj brings Gellar as a plus-one. After they are awarded their medals, Amy encourages girls to pursue science while Sheldon thanks his family and then, discarding the acceptance speech that he wrote as a child, individually acknowledges each of his friends and Amy as his other family who always support him, apologizing to all of them for not being the friend they deserved. In the last scene in the episode and the series, the gang is eating in Apartment 4A (an allusion to the final scene in the opening credits) with Sheldon and Amy wearing their medals as a melancholic acoustic version of the series' theme song's chorus plays. Title reference: Sheldon thinking Penny is ill on the flight to Stockholm; a reference to Stockholm syndrome."]
ep_data = [1,279]
df_corpus = pd.DataFrame({'text':lst_data,'ep':ep_data})

# Combine DataFrame into Dataset
from datasets import Dataset,Features,Value,ClassLabel

# Don't forget the class label data
class_names = [1,279]
ft = Features({'text': Value('string'), 'ep': ClassLabel(names=class_names)})

# Convert a single DataFrame to a Dataset
dataset_corpus = Dataset.from_pandas(df_corpus,features=ft)
dataset_corpus

# Tokenisation function
def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# apply to the entire dataset (train,test and validation dataset)
tokenised_dataset_corpus = dataset_corpus.map(tokenise, batched=True, batch_size=None)
print(tokenised_dataset_corpus.column_names)
# ['text', 'ep', 'input_ids', 'attention_mask']

# Change format to torch (input_ids -> tensor)
tokenised_dataset_corpus.set_format("torch",columns=["input_ids", "attention_mask", "ep"])
tokenised_dataset_corpus

def extract_hidden_states(batch):
    
    # Place model inputs on the GPU
    inputs = {k:v.to(device)
              for k,v in batch.items()
              if k in tokenizer.model_input_names}
    
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        
    # Return vector for [CLS] token (common practice)
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

# Extract last hidden states (faster w/ GPU)
dataset_corpus_hidden = tokenised_dataset_corpus.map(extract_hidden_states, batched=True)
dataset_corpus_hidden.column_names
print(dataset_corpus_hidden['hidden_state'])
print(dataset_corpus_hidden['hidden_state'].size())

# tensor([[-0.3574, -0.3052, -0.1603,  ...,  0.0195,  0.3460,  0.2879],
#        [-0.2919, -0.2997, -0.3504,  ...,  0.1250,  0.4126,  0.3579]])
# torch.Size([2, 768])
