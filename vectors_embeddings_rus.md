## Вектора и эмбединги для текстов

- Текст на естественном языке, который нужно обрабатывать в задачах машинного обучения сильно зависит от источника
- В связи с этим, возникает задача **предобработки** **нормализации** текста (приведения к некоторому единому виду)

Рассмотриим:
1. Приведение текста к нижнему регистру
2. Удаление неинформативных символов
3. Разбиение текста на смысловые единицы (токенизация)
4. Приведение слов к нормальной форме (стемминг/лемматизация)
5. Представление текста
6. Классификация

### 1 | Приведение текста к нижнему регистру

В базовом ритоне можно работать с `string`, `.lower()`, `.upper()` ...

```python
__add__ __class__ __contains__ __delattr__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __getitem__ 
__getnewargs__ __gt__ __hash__ __init__ __init_subclass__ __iter__ __le__ __len__ __lt__ __mod__ __mul__ __ne__ 
__new__ __reduce__ __reduce_ex__ __repr__ __rmod__ __rmul__ __setattr__ __sizeof__ __str__ __subclasshook__ 
capitalize casefold center count encode endswith expandtabs find format format_map index isalnum isalpha isascii 
isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper join ljust lower lstrip 
maketrans partition replace rfind rindex rjust rpartition rsplit rstrip split splitlines startswith strip 
swapcase title translate upper zfill 
```

```python
text = 'Способность эффективно обрабатывать данные с  \n большим числом признаков и классов... '
```

```python
text = text.lower()
text
```

```
'способность эффективно обрабатывать данные с  \n большим числом признаков и классов... '
```

### 2 | Удаление неинформативных символов

- Regular Expression <code>RE</code> можно использовать для этой задачи

```python
import re

regex = re.compile(r'(\W)\1+')
regex.sub(r'\1', text)
```

```
'способность эффективно обрабатывать данные с \n большим числом признаков и классов. '
```

```python
regex = re.compile(r'[^\w\s]')
text=regex.sub(r' ', text).strip()
```

```
'способность эффективно обрабатывать данные с  \n большим числом признаков и классов'
```

```python
re.sub('\s+', ' ', text)
```

```
'способность эффективно обрабатывать данные с большим числом признаков и классов'
```
### 3 | Разбиение текста на смысловые единицы (токенизация)

Самый простой подход к `токенизации`, это разбиение по текста по **пробельным** символам используя `.split`

Рассмотрим еще другие модули:

-  Модуль `pymorphy2`
-  Модуль `ntlk`
-  Модуль `razdel`
-  Модуль `spacy`

```python
text = 'Купите кружку-термос на 0.5л (64см³) за 3 рубля. До 01.01.2050.'
```
#### 3.1 | `split()` подход

По дефолту использует **" "** для разбиения <code>string</code>

```python
text.split()
```

```
['Купите',
 'кружку-термос',
 'на',
 '0.5л',
 '(64см³)',
 'за',
 '3',
 'рубля.',
 'До',
 '01.01.2050.']
 ```

#### 3.2 | Модуль `pymorphy2`

- В библиотеке для **морфологического анализа** для русского языка <code>pymorphy2</code>
- Простая вспомогательная функция `simple_word_tokenize` для `токенизации`

```python
GROUPING_SPACE_REGEX = re.compile(r'([^\w_-]|[+])', re.UNICODE)

def simple_word_tokenize(text, _split=GROUPING_SPACE_REGEX.split):
    """
    Split text into tokens. Don't split by a hyphen.
    Preserve punctuation, but not whitespaces.
    """
    return [t for t in _split(text) if t and not t.isspace()]
```

```python
from pymorphy2.tokenizers import simple_word_tokenize

simple_word_tokenize(text)

```

```
['Купите',
 'кружку-термос',
 'на',
 '0',
 '.',
 '5л',
 '(',
 '64см³',
 ')',
 'за',
 '3',
 'рубля',
 '.',
 'До',
 '01',
 '.',
 '01',
 '.',
 '2050',
 '.']
 ```
 
#### 3.3 | Модуль `ntlk`

- Более сложной метод <code>токенизации</code> представлен в `nltk`
- Используем метод `sent_tokenize`

```python
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize

sentences = sent_tokenize(text)
```

```python
[word_tokenize(sentence) for sentence in sentences]
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5л',
  '(',
  '64см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
 ```

```
[wordpunct_tokenize(sentence) for sentence in sentences]
```

```
[['Купите',
  'кружку',
  '-',
  'термос',
  'на',
  '0',
  '.',
  '5л',
  '(',
  '64см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01', '.', '01', '.', '2050', '.']]
 ```
 
#### 3.4 | Модуль `razdel`

Для Русского языка также есть новая специализированная библиотека <code>razdel</code>
 
 ```python
 import razdel

sents=[]
for sentence in razdel.sentenize(text):
    sents.append(sentence.text)
    
sents
```

```
['Купите кружку-термос на 0.5л (64см³) за 3 рубля.', 'До 01.01.2050.']
```

```python
sentences = [sentence.text for sentence in razdel.sentenize(text)]
tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5',
  'л',
  '(',
  '64',
  'см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
```

```python
razdel.tokenize
```

```
TokenSegmenter(TokenSplitter(),
               [DashRule(),
                UnderscoreRule(),
                FloatRule(),
                FractionRule(),
                FunctionRule('punct'),
                FunctionRule('other'),
                FunctionRule('yahoo')])
```

```python
import razdel

def tokenize_with_razdel(text):
    sentences = [sentence.text for sentence in razdel.sentenize(text)]
    tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
    
    return tokens

tokenize_with_razdel(text)
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5',
  'л',
  '(',
  '64',
  'см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
 ```
 
#### 3.5 | Модуль `SpaCy`

- `SpaCy` использует статистичестие модели, и включает в себя `токенизатока` в паиплайне
 
```python

text = 'Купите кружку-термос на 0.5л (64см³) за 3 рубля. До 01.01.2050.'

nlp = spacy.load("ru_core_news_sm")   # Более простая модель
# nlp = spacy.load("ru_core_news_lg") 

doc = nlp(text)

[token.text for token in doc] # tokenised text (stat model)
```

```
['Купите',
 'кружку',
 '-',
 'термос',
 'на',
 '0.5л',
 '(',
 '64',
 'см³',
 ')',
 'за',
 '3',
 'рубля',
 '.',
 'До',
 '01.01.2050',
 '.']
```
 
### 4. Приведение слов к нормальной форме (стемминг/лемматизация)

#### 4.1 Stemming

- <code>Стемминг</code> - это нормализация слова путём отбрасывания окончания по правилам языка
  - Такая нормализация хорошо подходит для языков с небольшим разнообразием словоформ, (английского)
  - В модуле <code>nltk</code> есть несколько реализаций стеммеров:
    - Porter stemmer
    - Snowball stemmer
    - Lancaster stemmer
  
```python
from nltk.stem.snowball import SnowballStemmer
SnowballStemmer(language='english').stem('running')
```

```
'run'
```
  
- Для Hусского языка этот подход не очень подходит, поскольку в русском есть:
  - **падежные формы**, **время у глаголов** и т.д.
    
```python
SnowballStemmer(language='russian').stem('бежать')
```

```
'бежа'
```

#### 4.2 | Lemmatisation
   
<code>Лемматизация</code> - приведение слов к начальной **морфологической форме** (с помощью **словаря** и **грамматики** языка)

 - Самый простой подход к лемматизации <code>словарный</code>.
 - Здесь не учитывается контекст слова, поэтому для омонимов такой подход работает не всегда.
 - Такой подход применяет библиотека <code>pymorphy2</code>

```
from pymorphy2 import MorphAnalyzer

pymorphy = MorphAnalyzer()
pymorphy.parse('бежал')
```

```
[Parse(word='бежал', tag=OpencorporaTag('VERB,perf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 1),)),
 Parse(word='бежал', tag=OpencorporaTag('VERB,impf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 49),))]
```

```python
def lemmatize_with_pymorphy(tokens):
    lemms = [pymorphy.parse(token)[0].normal_form for token in tokens]
    return lemms
```

```python
lemmatize_with_pymorphy(['бегут', 'бежал', 'бежите'])
```

```
['бежать', 'бежать', 'бежать']
```

```python
pymorphy.normal_forms('на заводе стали увидел виды стали')
```

```
['на заводе стали увидел виды стать', 'на заводе стали увидел виды сталь']
```

```python
lemmatize_with_pymorphy(['на', 'заводе', 'стали', 'увидел', 'виды', 'стали'])
```

```
['на', 'завод', 'стать', 'увидеть', 'вид', 'стать']
```

```python
pymorphy.parse('директора')
```

```
[Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc sing,gent'), normal_form='директор', score=0.632653, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 1),)),
 Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc plur,nomn'), normal_form='директор', score=0.204081, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 6),)),
 Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc sing,accs'), normal_form='директор', score=0.163265, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 3),))]
 ```
 
 - Модуль от Яндекса `mystem3` обходит это ограничение и **рассматривает контекст слова**, используя статистику и правила

```python
from pymystem3 import Mystem

mystem = Mystem()

def lemmatize_with_mystem(text):
    lemms=[token for token in mystem.lemmatize(text) if token!=' '][:-1]
    
    return  lemms
    
lemmatize_with_mystem('бегал бежал ')
```

```
['бегать', 'бежать']
```

```python
[token for token in mystem.lemmatize('бежал бежал') if token!=' '][:-1]
```

```
['бежать', 'бежать']
```

#### 5 | Представление Текста

Варианты представления текста можете назвать?

- Label Encoder
- One-Hot Encoder 
- Bag-of-Words
- TF-IDF

#### 5.1 | **Label Encoder**

Самый простой подход, используется с `padding` при подготовки матрицы

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

words = ['NLP', 'is', 'awesome']

label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(words)
```

```
array([0, 2, 1])
```

#### 5.2 | **One-Hot Encoding** (OHE)

```python
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(corpus_encoded.reshape(-1, 1))
```

```
array([[1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.]])
```

#### 5.3 | **Bag of Words**

- 4 текста, **один набор слов**, посчитаем количество слов в текстах <code>CountVectorizer</code>

```python
corpus = [
    'Девочка любит кота Ваську',
    'Тот кто любит, не знает кто любит его',
    'Кто кого любит?',
    'Васька любит девочка?',
]
```

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
vectors = vectorizer.fit_transform(corpus)
vectors.todense()
```

```
matrix([[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]])
```

```
vectorizer.vocabulary_
```

```
{'девочка': 2,
 'любит': 8,
 'кота': 6,
 'ваську': 1,
 'тот': 10,
 'кто': 7,
 'не': 9,
 'знает': 4,
 'его': 3,
 'кого': 5,
 'васька': 0}
```

#### 5.4 | **TF-IDF**

- **Term Frequency**  $tf(w,d)$ - сколько раз слово $w$ встретилось в документе $d$
- **Document Frequency** $df(w)$ - сколько документов содержат слово $w$
- **Inverse Document Frequency** $idf(w) = log_2(N/df(w))$  — обратная документная частотность. 
- **TF-IDF**=$tf(w,d)*idf(w)$

```python
from sklearn.feature_extraction.text import TfidfVectorizer
idf_vectorizer=TfidfVectorizer()
vectors = idf_vectorizer.fit_transform(corpus)
vectors.todense()
```

```
matrix([[0.        , 0.58783765, 0.46345796, 0.        , 0.        ,
         0.        , 0.58783765, 0.        , 0.30675807, 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , 0.36332075, 0.36332075,
         0.        , 0.        , 0.5728925 , 0.37919167, 0.36332075,
         0.36332075],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.72664149, 0.        , 0.5728925 , 0.37919167, 0.        ,
         0.        ],
        [0.72664149, 0.        , 0.5728925 , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.37919167, 0.        ,
         0.        ]])
```

```python
idf_vectorizer.vocabulary_
```

```
{'девочка': 2,
 'любит': 8,
 'кота': 6,
 'ваську': 1,
 'тот': 10,
 'кто': 7,
 'не': 9,
 'знает': 4,
 'его': 3,
 'кого': 5,
 'васька': 0}
```

### 6 | Классификация

- Мы применим методы **предобработки** и **представления текста** на примере анализа тональности текста
- В качестве данных, используем датасет твитов. Всего в данных 2 класса, позитив и негативное мнение

#### 6.1 | Загрузка тренировочных и тестовых данных

- Данные загружены в папку [<code>data/embeddings/</code>](https://github.com/shtrausslearning/nlp/tree/main/data/embeddings)

```python

import pandas as pd

train = pd.read_csv('kaggle/input/embeddings/train.csv')
test = pd.read_csv('kaggle/input/embeddings/test.csv')
train.head()
```

```
|    | label    | text                                                                                                                                          |
|---:|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | positive | эти розы для прекрасной мамочки)))=_=]]                                                                                                       |
|  1 | negative | И да, у меня в этом году серьезные проблемы со сном и режимом.                                                                                |
|  2 | positive | Обожаю людей, которые заставляют меня смеяться.©                                                                                           |
|  3 | negative | Вчера нашла в почтовом ящике пустую упаковку из-под жвачки и использованный презерватив в упаковке. Могли бы хоть "резинку" не портить, гады. |
|  4 | positive | очень долгожданный и хороший день был)                                                                                                        |
```

- Распределение классов

```python
train.label.value_counts(normalize=True)
```

```
positive    0.668928
negative    0.331072
Name: label, dtype: float64
```

#### 6.2 | Построение модели

- Напишем функцию для оценки **векторизатора**
- В качестве модели будем использовать **линейный SVM**, он хорошо работает для определения тональности

```python

%matplotlib inline

import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Создание модели классификации
def evaluate_vectoriser(vectoriser):

    # Preprocess
    train_vectors = vectoriser.fit_transform(train['text'])
    test_vectors = vectoriser.transform(test['text'])
    
    # Train & Predict
    clf = LinearSVC(random_state=42)
    clf.fit(train_vectors, train['label'])
    predictions = clf.predict(test_vectors)
    
    # Classification Report
    print(classification_report(test['label'], predictions))
    
    return predictions
  
```

#### 6.3 | Сравнение способов представления текста

Для решения задачи, рассмотрим разные подходы:

- `CountVectorizer`
- `TfidfVectorizer`
- `TfidfVectorizer` + `tokenizer=tokenize_with_razdel`
- `TfidfVectorizer` + `lemmatize_with_pymorphy(tokenize_with_razdel(text)`
- `TfidfVectorizer` + `tokenizer` + `stop_words`
- `TfidfVectorizer` + `tokenizer` + `stop_words` + `ngram_range`

#### Подход 1 : `CountVectorizer`

```python
from sklearn.feature_extraction.text import CountVectorizer

evaluate_vectorizer(CountVectorizer(min_df=2));
```

```
              precision    recall  f1-score   support

    negative       0.73      0.62      0.67       258
    positive       0.83      0.89      0.86       536

    accuracy                           0.80       794
   macro avg       0.78      0.75      0.76       794
weighted avg       0.80      0.80      0.80       794
```

#### Подход 2 : `TfidfVectorizer`
  
```python
from sklearn.feature_extraction.text import TfidfVectorize

evaluate_vectoriser(TfidfVectorizer(min_df=2));
```

```
              precision    recall  f1-score   support

    negative       0.76      0.65      0.70       258
    positive       0.84      0.90      0.87       536

    accuracy                           0.82       794
   macro avg       0.80      0.78      0.79       794
weighted avg       0.82      0.82      0.82       794
```

#### Подход 3 : `TfidfVectorizer` + `tokenizer=tokenize_with_razdel`

```python
def tokenize_with_razdel(text):
    return [token.text for token in razdel.tokenize(text)] # tokens
```

```python
evaluate_vectoriser(TfidfVectorizer(min_df=2,
                                    tokenizer=tokenize_with_razdel));
```

#### Подход 4 : `TfidfVectorizer` + `lemmatize_with_pymorphy(tokenize_with_razdel(text)`

```python
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
)

predictions=evaluate_vectorizer(tfidf_vectorizer)
```

```
              precision    recall  f1-score   support

    negative       0.85      0.79      0.82       258
    positive       0.90      0.93      0.92       536

    accuracy                           0.89       794
   macro avg       0.88      0.86      0.87       794
weighted avg       0.89      0.89      0.89       794
```


#### Подход 5 : `TfidfVectorizer` + `tokenizer` + `stop_words`

```python
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
    stop_words=stopwords
)

evaluate_vectorizer(tfidf_vectorizer)
```

```
              precision    recall  f1-score   support

    negative       0.83      0.75      0.79       258
    positive       0.89      0.93      0.91       536

    accuracy                           0.87       794
   macro avg       0.86      0.84      0.85       794
weighted avg       0.87      0.87      0.87       794
```

#### Подход 6 : `TfidfVectorizer` + `tokenizer` + `stop_words` + `ngram_range`

```python
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
    stop_words=stopwords,
    ngram_range=(1, 2)
)

evaluate_vectorizer(tfidf_vectorizer)
```

```
              precision    recall  f1-score   support

    negative       0.84      0.78      0.81       258
    positive       0.90      0.93      0.91       536

    accuracy                           0.88       794
   macro avg       0.87      0.85      0.86       794
weighted avg       0.88      0.88      0.88       794
```
