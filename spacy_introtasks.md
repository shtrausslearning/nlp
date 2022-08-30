
### Tokenisation

```python

import spacy

# load small statistical model
nlp = spacy.load("en_core_web_sm")

# Tokenization on a simple string
sentence = nlp.tokenizer("We live in Paris.")

# Length of sentence
print("The number of tokens: ", len(sentence))

# Print individual words (i.e., tokens)
print("The tokens: ")
for words in sentence:
    print(words)
    
```

```
The number of tokens:  5
The tokens: 
We
live
in
Paris
.
```

```python
import pandas as pd
import os
cwd = os.getcwd()

# Import Jeopardy Questions
data = pd.read_csv(cwd+'/JEOPARDY_CSV.csv')
data = pd.DataFrame(data)

# Lowercase, strip whitespace, and view column names
data.columns = map(lambda x: x.lower().strip(), data.columns)

# Select only a subset of data
data = data[0:100]

# Tokenize Jeopardy Questions
data["question_tkn"] = data["question"].apply(lambda x: nlp(x))

```

```
0     (For, the, last, 8, years, of, his, life, ,, G...
1     (No, ., 2, :, 1912, Olympian, ;, football, sta...
2     (The, city, of, Yuma, in, this, state, has, a,...
3     (In, 1963, ,, live, on, ", The, Art, Linklette...
4     (Signer, of, the, Dec., of, Indep, ., ,, frame...
                            ...                        
95    (Say, <, a, href="http://www.j, -, archive.com...
96    (This, car, company, has, been, in, the, news,...
97    (As, an, adjective, ,, it, can, mean, proper, ...
98    (The, wedge, is, an, adaptation, of, the, simp...
99    (With, a, mighty, leap, of, 5'1, ", ,, David, ...
Name: question_tkn, Length: 100, dtype: object
```

### POS tagging

Machines need to tag each token with relevant metadata, such as the part-of-speech of each token

```python

# View first question
example_question = data.question[0]
example_question_tokens = data.question_tkn[0]
print("The first questions is:")
print(example_question,'\n')    
```

```
The first questions is:
For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory 
```

```python

# Print Part-of-speech tags for tokens in the first question
print("Here are the Part-of-speech tags for each token in the first question:")
for token in example_question_tokens:
    print(token.text,token.pos_, spacy.explain(token.pos_))
```

```
Here are the Part-of-speech tags for each token in the first question:
For ADP adposition
the DET determiner
last ADJ adjective
8 NUM numeral
years NOUN noun
of ADP adposition
his PRON pronoun
life NOUN noun
, PUNCT punctuation
Galileo PROPN proper noun
was AUX auxiliary
under ADP adposition
house NOUN noun
arrest NOUN noun
for ADP adposition
espousing VERB verb
this DET determiner
man NOUN noun
's PART particle
theory NOUN noun
```

### Dependency Parsing

- Dependency parsing is the process of finding these relationships among the tokens. 
- Once we have performed this step, we will be able to visualize the relationships using a dependency parsing graph.

```python
for token in example_question_tokens:
    print(token.text,token.dep_, spacy.explain(token.dep_))
```

```
For prep prepositional modifier
the det determiner
last amod adjectival modifier
8 nummod numeric modifier
years pobj object of preposition
of prep prepositional modifier
his poss possession modifier
life pobj object of preposition
, punct punctuation
Galileo nsubj nominal subject
was ROOT root
under prep prepositional modifier
house compound compound
arrest pobj object of preposition
for prep prepositional modifier
espousing pcomp complement of preposition
this det determiner
man poss possession modifier
's case case marking
theory dobj direct object
```

```python
# Visualize the dependency parse
from spacy import displacy
from pathlib import Path

svg = displacy.render(example_question_tokens, 
                      style='dep',
                      jupyter=False, 
                      options={'distance': 90})

output_path = Path("dependency_plot.svg")
output_path.open("w", encoding="utf-8").write(svg)
```

![](https://raw.githubusercontent.com/shtrausslearning/nlp/7f4a929e1f586547cf1183618ef70c1a7c89e993/data/dependency_plot.svg)
    
