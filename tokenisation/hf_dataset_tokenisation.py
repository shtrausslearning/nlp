''' Prepare Data '''

# Pandas DataFrame to HF Dataset
from datasets import Dataset,Features,Value,ClassLabel
import pandas as pd

lst_data = ["After an unsuccessful visit to the high-IQ sperm bank, Dr. Leonard Hofstadter and Dr. Sheldon Cooper return home to find aspiring actress Penny is their new neighbor across the hall from their apartment. Sheldon thinks Leonard, who is immediately interested in her, is chasing a dream he will never catch. Leonard invites Penny to his and Sheldon's apartment for Indian food, where she asks to use their shower since hers is broken. While wrapped in a towel, she gets to meet their visiting friends Howard Wolowitz, a wannabe ladies' man who tries to hit on her, and Rajesh Koothrappali, who is unable to speak to her as he suffers from selective mutism in the presence of women. Leonard is so infatuated with Penny that, after helping her use their shower, he agrees to retrieve her TV from her ex-boyfriend Kurt. However, Kurt's physical superiority overwhelms Leonard's and Sheldon's combined IQ of 360, and they return without pants or TV. Penny, feeling bad, offers to take the guys out to dinner, initiating a friendship with them.",
            "After Sheldon and Leonard spend two months repairing Sheldon's DNA molecule model, everyone prepares to fly to Sweden for the Nobel Prize award ceremony. Howard and Bernadette nervously leave their kids for the first time with Stuart and Denise, while Raj leaves his dog with Bert. Penny has become pregnant, though she and Leonard are keeping it a secret. On the flight, Raj meets Sarah Michelle Gellar. Penny's frequent bathroom trips make Sheldon fear she is sick. Penny reveals her pregnancy to Sheldon but, instead of being excited for her, Sheldon is only selfishly relieved that he will not get sick, and he exposes the pregnancy, offending Leonard. At the hotel, a series of minor incidents with their kids make Howard and Bernadette want to go home. Much to their dismay, Sheldon is still insensitive. Amy furiously tells Sheldon he broke his friends' hearts and that people (sometimes including her) only tolerate him because he does not intentionally do so. Everyone decides to stay for the ceremony and Raj brings Gellar as a plus-one. After they are awarded their medals, Amy encourages girls to pursue science while Sheldon thanks his family and then, discarding the acceptance speech that he wrote as a child, individually acknowledges each of his friends and Amy as his other family who always support him, apologizing to all of them for not being the friend they deserved. In the last scene in the episode and the series, the gang is eating in Apartment 4A (an allusion to the final scene in the opening credits) with Sheldon and Amy wearing their medals as a melancholic acoustic version of the series' theme song's chorus plays. Title reference: Sheldon thinking Penny is ill on the flight to Stockholm; a reference to Stockholm syndrome."]
ep_data = [1,279]
df_corpus = pd.DataFrame({'text':lst_data,'ep':ep_data})

# Don't forget the class label data
class_names = [1,279]
ft = Features({'text': Value('string'), 'ep': ClassLabel(names=class_names)})

# Convert a single DataFrame to a Dataset
dataset_corpus = Dataset.from_pandas(df_corpus,features=ft)

''' Tokenisation '''

from transformers import AutoTokenizer

''' Working w/ Tokeniser (list) '''
print('Corpus documents:')
for document in dataset_corpus['text']:
    print(document)

# AutoTokeniser Subword Tokenisation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenisation function
def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# apply to the entire dataset (train,test and validation dataset)
encoded_corpus = dataset_corpus.map(tokenise, batched=True, batch_size=None)
print(encoded_corpus.column_names)

print(f'\ntokenised document:\n',encoded_corpus['input_ids'])            # encoded values
print(f'\nattention mask document:\n',encoded_corpus['attention_mask'],'\n')  # encoded values mask (for multiple documents)

# tokeniser information (general)
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Tokeniser model input names: {tokenizer.model_input_names}','\n')

# Convert tokenised numeric values to words from dictionary
lst_tokens = []
for ii,corpus in enumerate(encoded_corpus['input_ids']):
    tokens = tokenizer.convert_ids_to_tokens(encoded_corpus['input_ids'][ii])
    lst_tokens.append(tokens)
    print(tokens)

print('\nConvert tokens to string')
print(tokenizer.convert_tokens_to_string(tokens),'\n')

# Corpus documents:
# After an unsuccessful visit to the high-IQ sperm bank, Dr. Leonard Hofstadter and Dr. Sheldon Cooper return home to find aspiring actress Penny is their new neighbor across the hall from their apartment. Sheldon thinks Leonard, who is immediately interested in her, is chasing a dream he will never catch. Leonard invites Penny to his and Sheldon's apartment for Indian food, where she asks to use their shower since hers is broken. While wrapped in a towel, she gets to meet their visiting friends Howard Wolowitz, a wannabe ladies' man who tries to hit on her, and Rajesh Koothrappali, who is unable to speak to her as he suffers from selective mutism in the presence of women. Leonard is so infatuated with Penny that, after helping her use their shower, he agrees to retrieve her TV from her ex-boyfriend Kurt. However, Kurt's physical superiority overwhelms Leonard's and Sheldon's combined IQ of 360, and they return without pants or TV. Penny, feeling bad, offers to take the guys out to dinner, initiating a friendship with them.
# After Sheldon and Leonard spend two months repairing Sheldon's DNA molecule model, everyone prepares to fly to Sweden for the Nobel Prize award ceremony. Howard and Bernadette nervously leave their kids for the first time with Stuart and Denise, while Raj leaves his dog with Bert. Penny has become pregnant, though she and Leonard are keeping it a secret. On the flight, Raj meets Sarah Michelle Gellar. Penny's frequent bathroom trips make Sheldon fear she is sick. Penny reveals her pregnancy to Sheldon but, instead of being excited for her, Sheldon is only selfishly relieved that he will not get sick, and he exposes the pregnancy, offending Leonard. At the hotel, a series of minor incidents with their kids make Howard and Bernadette want to go home. Much to their dismay, Sheldon is still insensitive. Amy furiously tells Sheldon he broke his friends' hearts and that people (sometimes including her) only tolerate him because he does not intentionally do so. Everyone decides to stay for the ceremony and Raj brings Gellar as a plus-one. After they are awarded their medals, Amy encourages girls to pursue science while Sheldon thanks his family and then, discarding the acceptance speech that he wrote as a child, individually acknowledges each of his friends and Amy as his other family who always support him, apologizing to all of them for not being the friend they deserved. In the last scene in the episode and the series, the gang is eating in Apartment 4A (an allusion to the final scene in the opening credits) with Sheldon and Amy wearing their medals as a melancholic acoustic version of the series' theme song's chorus plays. Title reference: Sheldon thinking Penny is ill on the flight to Stockholm; a reference to Stockholm syndrome.

# ['text', 'ep', 'input_ids', 'attention_mask']

# tokenised document:
# [[101, 2044, 2019, 7736, 3942, 2000, 1996, 2152, 1011, 26264, 18047, 2924, 1010, 2852, 1012, 7723, 7570, 10343, 18808, 2121, 1998, 2852, 1012, 19369, 6201, 2709, 2188, 2000, 2424, 22344, 3883, 10647, 2003, 2037, 2047, 11429, 2408, 1996, 2534, 2013, 2037, 4545, 1012, 19369, 6732, 7723, 1010, 2040, 2003, 3202, 4699, 1999, 2014, 1010, 2003, 11777, 1037, 3959, 2002, 2097, 2196, 4608, 1012, 7723, 18675, 10647, 2000, 2010, 1998, 19369, 1005, 1055, 4545, 2005, 2796, 2833, 1010, 2073, 2016, 5176, 2000, 2224, 2037, 6457, 2144, 5106, 2003, 3714, 1012, 2096, 5058, 1999, 1037, 10257, 1010, 2016, 4152, 2000, 3113, 2037, 5873, 2814, 4922, 24185, 8261, 8838, 1010, 1037, 10587, 4783, 6456, 1005, 2158, 2040, 5363, 2000, 2718, 2006, 2014, 1010, 1998, 11948, 9953, 12849, 14573, 2527, 13944, 3669, 1010, 2040, 2003, 4039, 2000, 3713, 2000, 2014, 2004, 2002, 17567, 2013, 13228, 14163, 17456, 1999, 1996, 3739, 1997, 2308, 1012, 7723, 2003, 2061, 1999, 27753, 16453, 2007, 10647, 2008, 1010, 2044, 5094, 2014, 2224, 2037, 6457, 1010, 2002, 10217, 2000, 12850, 2014, 2694, 2013, 2014, 4654, 1011, 6898, 9679, 1012, 2174, 1010, 9679, 1005, 1055, 3558, 19113, 2058, 2860, 24546, 2015, 7723, 1005, 1055, 1998, 19369, 1005, 1055, 4117, 26264, 1997, 9475, 1010, 1998, 2027, 2709, 2302, 6471, 2030, 2694, 1012, 10647, 1010, 3110, 2919, 1010, 4107, 2000, 2202, 1996, 4364, 2041, 2000, 4596, 1010, 26616, 1037, 6860, 2007, 2068, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2044, 19369, 1998, 7723, 5247, 2048, 2706, 26296, 19369, 1005, 1055, 6064, 13922, 2944, 1010, 3071, 20776, 2000, 4875, 2000, 4701, 2005, 1996, 10501, 3396, 2400, 5103, 1012, 4922, 1998, 16595, 9648, 4674, 12531, 2681, 2037, 4268, 2005, 1996, 2034, 2051, 2007, 6990, 1998, 15339, 1010, 2096, 11948, 3727, 2010, 3899, 2007, 14324, 1012, 10647, 2038, 2468, 6875, 1010, 2295, 2016, 1998, 7723, 2024, 4363, 2009, 1037, 3595, 1012, 2006, 1996, 3462, 1010, 11948, 6010, 4532, 9393, 21500, 8017, 1012, 10647, 1005, 1055, 6976, 5723, 9109, 2191, 19369, 3571, 2016, 2003, 5305, 1012, 10647, 7657, 2014, 10032, 2000, 19369, 2021, 1010, 2612, 1997, 2108, 7568, 2005, 2014, 1010, 19369, 2003, 2069, 14337, 2135, 7653, 2008, 2002, 2097, 2025, 2131, 5305, 1010, 1998, 2002, 14451, 2015, 1996, 10032, 1010, 2125, 18537, 7723, 1012, 2012, 1996, 3309, 1010, 1037, 2186, 1997, 3576, 10444, 2007, 2037, 4268, 2191, 4922, 1998, 16595, 9648, 4674, 2215, 2000, 2175, 2188, 1012, 2172, 2000, 2037, 20006, 1010, 19369, 2003, 2145, 16021, 6132, 13043, 1012, 6864, 20322, 4136, 19369, 2002, 3631, 2010, 2814, 1005, 8072, 1998, 2008, 2111, 1006, 2823, 2164, 2014, 1007, 2069, 19242, 2032, 2138, 2002, 2515, 2025, 15734, 2079, 2061, 1012, 3071, 7288, 2000, 2994, 2005, 1996, 5103, 1998, 11948, 7545, 21500, 8017, 2004, 1037, 4606, 1011, 2028, 1012, 2044, 2027, 2024, 3018, 2037, 6665, 1010, 6864, 16171, 3057, 2000, 7323, 2671, 2096, 19369, 4283, 2010, 2155, 1998, 2059, 1010, 5860, 29154, 1996, 9920, 4613, 2008, 2002, 2626, 2004, 1037, 2775, 1010, 14258, 28049, 2169, 1997, 2010, 2814, 1998, 6864, 2004, 2010, 2060, 2155, 2040, 2467, 2490, 2032, 1010, 9706, 12898, 28660, 2000, 2035, 1997, 2068, 2005, 2025, 2108, 1996, 2767, 2027, 10849, 1012, 1999, 1996, 2197, 3496, 1999, 1996, 2792, 1998, 1996, 2186, 1010, 1996, 6080, 2003, 5983, 1999, 4545, 26424, 1006, 2019, 2035, 14499, 2000, 1996, 2345, 3496, 1999, 1996, 3098, 6495, 1007, 2007, 19369, 1998, 6864, 4147, 2037, 6665, 2004, 1037, 11463, 2319, 9905, 10415, 6490, 2544, 1997, 1996, 2186, 1005, 4323, 2299, 1005, 1055, 7165, 3248, 1012, 2516, 4431, 1024, 19369, 3241, 10647, 2003, 5665, 2006, 1996, 3462, 2000, 8947, 1025, 1037, 4431, 2000, 8947, 8715, 1012, 102]]

# attention mask document:
# [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 

# Vocab size: 30522
# Max length: 512
# Tokeniser model input names: ['input_ids', 'attention_mask'] 

# ['[CLS]', 'after', 'an', 'unsuccessful', 'visit', 'to', 'the', 'high', '-', 'iq', 'sperm', 'bank', ',', 'dr', '.', 'leonard', 'ho', '##fs', '##tadt', '##er', 'and', 'dr', '.', 'sheldon', 'cooper', 'return', 'home', 'to', 'find', 'aspiring', 'actress', 'penny', 'is', 'their', 'new', 'neighbor', 'across', 'the', 'hall', 'from', 'their', 'apartment', '.', 'sheldon', 'thinks', 'leonard', ',', 'who', 'is', 'immediately', 'interested', 'in', 'her', ',', 'is', 'chasing', 'a', 'dream', 'he', 'will', 'never', 'catch', '.', 'leonard', 'invites', 'penny', 'to', 'his', 'and', 'sheldon', "'", 's', 'apartment', 'for', 'indian', 'food', ',', 'where', 'she', 'asks', 'to', 'use', 'their', 'shower', 'since', 'hers', 'is', 'broken', '.', 'while', 'wrapped', 'in', 'a', 'towel', ',', 'she', 'gets', 'to', 'meet', 'their', 'visiting', 'friends', 'howard', 'wo', '##low', '##itz', ',', 'a', 'wanna', '##be', 'ladies', "'", 'man', 'who', 'tries', 'to', 'hit', 'on', 'her', ',', 'and', 'raj', '##esh', 'ko', '##oth', '##ra', '##ppa', '##li', ',', 'who', 'is', 'unable', 'to', 'speak', 'to', 'her', 'as', 'he', 'suffers', 'from', 'selective', 'mu', '##tism', 'in', 'the', 'presence', 'of', 'women', '.', 'leonard', 'is', 'so', 'in', '##fat', '##uated', 'with', 'penny', 'that', ',', 'after', 'helping', 'her', 'use', 'their', 'shower', ',', 'he', 'agrees', 'to', 'retrieve', 'her', 'tv', 'from', 'her', 'ex', '-', 'boyfriend', 'kurt', '.', 'however', ',', 'kurt', "'", 's', 'physical', 'superiority', 'over', '##w', '##helm', '##s', 'leonard', "'", 's', 'and', 'sheldon', "'", 's', 'combined', 'iq', 'of', '360', ',', 'and', 'they', 'return', 'without', 'pants', 'or', 'tv', '.', 'penny', ',', 'feeling', 'bad', ',', 'offers', 'to', 'take', 'the', 'guys', 'out', 'to', 'dinner', ',', 'initiating', 'a', 'friendship', 'with', 'them', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# ['[CLS]', 'after', 'sheldon', 'and', 'leonard', 'spend', 'two', 'months', 'repairing', 'sheldon', "'", 's', 'dna', 'molecule', 'model', ',', 'everyone', 'prepares', 'to', 'fly', 'to', 'sweden', 'for', 'the', 'nobel', 'prize', 'award', 'ceremony', '.', 'howard', 'and', 'bern', '##ade', '##tte', 'nervously', 'leave', 'their', 'kids', 'for', 'the', 'first', 'time', 'with', 'stuart', 'and', 'denise', ',', 'while', 'raj', 'leaves', 'his', 'dog', 'with', 'bert', '.', 'penny', 'has', 'become', 'pregnant', ',', 'though', 'she', 'and', 'leonard', 'are', 'keeping', 'it', 'a', 'secret', '.', 'on', 'the', 'flight', ',', 'raj', 'meets', 'sarah', 'michelle', 'gel', '##lar', '.', 'penny', "'", 's', 'frequent', 'bathroom', 'trips', 'make', 'sheldon', 'fear', 'she', 'is', 'sick', '.', 'penny', 'reveals', 'her', 'pregnancy', 'to', 'sheldon', 'but', ',', 'instead', 'of', 'being', 'excited', 'for', 'her', ',', 'sheldon', 'is', 'only', 'selfish', '##ly', 'relieved', 'that', 'he', 'will', 'not', 'get', 'sick', ',', 'and', 'he', 'expose', '##s', 'the', 'pregnancy', ',', 'off', '##ending', 'leonard', '.', 'at', 'the', 'hotel', ',', 'a', 'series', 'of', 'minor', 'incidents', 'with', 'their', 'kids', 'make', 'howard', 'and', 'bern', '##ade', '##tte', 'want', 'to', 'go', 'home', '.', 'much', 'to', 'their', 'dismay', ',', 'sheldon', 'is', 'still', 'ins', '##ens', '##itive', '.', 'amy', 'furiously', 'tells', 'sheldon', 'he', 'broke', 'his', 'friends', "'", 'hearts', 'and', 'that', 'people', '(', 'sometimes', 'including', 'her', ')', 'only', 'tolerate', 'him', 'because', 'he', 'does', 'not', 'intentionally', 'do', 'so', '.', 'everyone', 'decides', 'to', 'stay', 'for', 'the', 'ceremony', 'and', 'raj', 'brings', 'gel', '##lar', 'as', 'a', 'plus', '-', 'one', '.', 'after', 'they', 'are', 'awarded', 'their', 'medals', ',', 'amy', 'encourages', 'girls', 'to', 'pursue', 'science', 'while', 'sheldon', 'thanks', 'his', 'family', 'and', 'then', ',', 'disc', '##arding', 'the', 'acceptance', 'speech', 'that', 'he', 'wrote', 'as', 'a', 'child', ',', 'individually', 'acknowledges', 'each', 'of', 'his', 'friends', 'and', 'amy', 'as', 'his', 'other', 'family', 'who', 'always', 'support', 'him', ',', 'ap', '##olo', '##gizing', 'to', 'all', 'of', 'them', 'for', 'not', 'being', 'the', 'friend', 'they', 'deserved', '.', 'in', 'the', 'last', 'scene', 'in', 'the', 'episode', 'and', 'the', 'series', ',', 'the', 'gang', 'is', 'eating', 'in', 'apartment', '4a', '(', 'an', 'all', '##usion', 'to', 'the', 'final', 'scene', 'in', 'the', 'opening', 'credits', ')', 'with', 'sheldon', 'and', 'amy', 'wearing', 'their', 'medals', 'as', 'a', 'mel', '##an', '##cho', '##lic', 'acoustic', 'version', 'of', 'the', 'series', "'", 'theme', 'song', "'", 's', 'chorus', 'plays', '.', 'title', 'reference', ':', 'sheldon', 'thinking', 'penny', 'is', 'ill', 'on', 'the', 'flight', 'to', 'stockholm', ';', 'a', 'reference', 'to', 'stockholm', 'syndrome', '.', '[SEP]']

# Convert tokens to string
# [CLS] after sheldon and leonard spend two months repairing sheldon's dna molecule model, everyone prepares to fly to sweden for the nobel prize award ceremony. howard and bernadette nervously leave their kids for the first time with stuart and denise, while raj leaves his dog with bert. penny has become pregnant, though she and leonard are keeping it a secret. on the flight, raj meets sarah michelle gellar. penny's frequent bathroom trips make sheldon fear she is sick. penny reveals her pregnancy to sheldon but, instead of being excited for her, sheldon is only selfishly relieved that he will not get sick, and he exposes the pregnancy, offending leonard. at the hotel, a series of minor incidents with their kids make howard and bernadette want to go home. much to their dismay, sheldon is still insensitive. amy furiously tells sheldon he broke his friends'hearts and that people ( sometimes including her ) only tolerate him because he does not intentionally do so. everyone decides to stay for the ceremony and raj brings gellar as a plus - one. after they are awarded their medals, amy encourages girls to pursue science while sheldon thanks his family and then, discarding the acceptance speech that he wrote as a child, individually acknowledges each of his friends and amy as his other family who always support him, apologizing to all of them for not being the friend they deserved. in the last scene in the episode and the series, the gang is eating in apartment 4a ( an allusion to the final scene in the opening credits ) with sheldon and amy wearing their medals as a melancholic acoustic version of the series'theme song's chorus plays. title reference : sheldon thinking penny is ill on the flight to stockholm ; a reference to stockholm syndrome. [SEP] 