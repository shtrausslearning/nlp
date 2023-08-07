import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class dbRetreiver():

    def __init__(self, df, **kwargs):

        self.df = df
        self.vectoriser = TfidfVectorizer(**kwargs)
        self.vectoriser.fit(self.df['question'])
        self.vectors = self.vectoriser.transform(self.df['question'])

    # transform TFIDF on new text data

    def _q_idx(self, question:str):
        vector = self.vectoriser.transform([question])

        # dot product of two vectors (work fine)
        similarities = vector.dot(self.vectors.T) 
        return similarities.argmax()

    # respond to user request

    def reply(self, question:str):
        idx = self._q_idx(question)
        self.df['question'][idx]
        response = (
            f"Request:\n {question}\n"
            f"Found Question:\n {self.df['question'][idx]}\n"
            f"Answer to Question:\n {self.df['answer'][idx]}\n"
        )
        return response


'''

Run Bot

'''

SAMPLE_PATH = 'sample_faq.csv'

def run_bot(db=None):

    # if db not provided load sample
    if db is None:
        df = pd.read_csv(SAMPLE_PATH)
        faq = FAQ(df)

    print('Your request: ')
    question = input('>> ')

    # request loop

    log = []
    while True:
        if question:
            if question.lower().strip().startswith('exit'):
                break
            answer = db.reply(question)
            log.append([question, answer])
            print(answer)

        print('Your request: ')
        question = input('>> ')
        if('quit' in question):
            return log           # return 
            break                # break out


# load any data
df = pd.read_csv('sample_faq.csv')

# retreive similar answer from db
faq = dbRetreiver(df)
print(faq.reply('what are the data types in python?'))

# chatbot (ask questions) & return log upon quit
qa_log = run_bot(faq)
