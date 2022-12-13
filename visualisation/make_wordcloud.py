import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain

# Example Text
text = '''

With Leonard, Howard, Raj, and Amy accomplishing so much on their respective projects, Sheldon is forced to admit he has nothing important upon which to work. 
He makes Amy leave the apartment for a few days so he can focus, but cannot come up with any ideas and calls his mother as a distraction. 
Leonard and Amy have fun recreating experiments from when they were growing up, boring Penny, so she eats with Sheldon as he mulls over his scientific studies. 
Penny helps him realize that his study of dark matter is his rebound science from string theory, which Sheldon admits he never truly disregarded, 
but explaining string theory to her inspires Sheldon, helping him discover a potential breakthrough in the field. 
Meanwhile, Howard is too busy with his family to be in the band with Raj, so Raj brings in Bert. 
But when Howard annoys Bernadette by writing an astronaut-themed musical while she is on bed rest, she makes him rejoin the band. 
The three are poorly received at a Bar mitzvah after singing Bert's original song about the boulder from Raiders of the Lost Ark. 
Title reference: A triple entendre to represent Sheldon going into isolation to figure out his future research field only to go back to studying string theory, 
the vibration of the strings in string theory, and Howard's oscillation between being in a band with Raj and being solo.

'''

# Load statistical model
nlp = spacy.load('en_core_web_sm')

# List of documents (just string text)
lst_documents = [text,text]

# Tokenise list of documents
lst_issue = []
for text in lst_documents:
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lst_issue.append(tokens)
    
# Combine lists
all = list(chain(*lst_issue))  
text = " ".join(word for word in all)

# Make word cloud
plt.figure(figsize=(20, 15))
word_cloud1 = WordCloud(collocations = True, background_color = 'white',
                        width = 2000, height = 1080).generate(text)
plt.imshow(word_cloud1)
plt.axis("off")
plt.tight_layout()
plt.show()
