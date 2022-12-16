
''' 

Converting DataFrame to Dataset 

'''
# Have some dataframe containing text & labels
# data isn't necessarily compatible, just demonstration

# Split the data into subsets
from sklearn.model_selection import train_test_split as tts

train_files,test_files, train_labels, test_labels = tts(df_data['text'],
                                                        df_data['label'],
                                                        test_size=0.2,
                                                        random_state=32,
                                                        stratify=df_data['label'])

train_files = pd.DataFrame(train_files)
test_files = pd.DataFrame(test_files)
train_files['label'] = train_labels
test_files['label'] = test_labels

print(train_files['label'].value_counts(),'\n')
print(test_files['label'].value_counts())

# [approach 1] Using DatasetDict combine the DataFrame data
from datasets import Dataset,DatasetDict,Features,Value,ClassLabel

# Don't forget the class label data
class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
ft = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

# Combine Multiple Datasets 
emotions = DatasetDict({
    "train": Dataset.from_pandas(train,features=ft),
    "test": Dataset.from_pandas(test,features=ft),
    "validation": Dataset.from_pandas(validation,features=ft)
    })

# [approach 2] Using Dataset
# class_encode_column -> labelencode string column

traindts = Dataset.from_pandas(train_files)
traindts = traindts.class_encode_column("label")
testdts = Dataset.from_pandas(test_files)
testdts = testdts.class_encode_column("label") 

# Pandas indicies not reset ie. __index_level_0__ additional column
corpus = DatasetDict({"train" : traindts , "validation" : testdts })
corpus['train']

'''

Change representation of Dataset to DataFrame

'''
# working with Dataset, we may need to convert the data to DataFrames (plots etc)

corpus.set_format(type="pandas")
ldf = corpus["train"][:]

# Add label data to dataframe
def label_int2str(row):
    return corpus["train"].features["label"].int2str(row)

ldf["label_name"] = ldf["label"].apply(label_int2str)

# When done 
corpus.reset_format()

