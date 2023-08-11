from typing import List
import regex as re

'''

PARSER FOR THE DATASET NER TAG FORMAT

'''

class Parser:
    
    # RE patterns for tag extraction
    LABEL_PATTERN = r"\[(.*?)\]"
    PUNCTUATION_PATTERN = r"([.,\/#!$%\^&\*;:{}=\-_`~()'\"’¿])"
    
    # initialise, first word/id tag is O (outside)
    def __init__(self):
        self.tag_to_id = {
            "O": 0
        }
        self.id_to_tag = {
            0: "O"
        }
        
    '''
    
    CREATE TAGS
    
    '''
    # input : sentence, tagged sentence
    
    def __call__(self, sentence: str, annotated: str) -> List[str]:
    
        ''' Create Dictionary of Identified Tags'''
    
        # 1. set label B or I    
        
        matches = re.findall(self.LABEL_PATTERN, annotated)
        word_to_tag = {}
        for match in matches:
            tag, phrase = match.split(" : ")
            words = phrase.split(" ") 
            word_to_tag[words[0]] = f"B-{tag.upper()}"
            for w in words[1:]:
                word_to_tag[w] = f"I-{tag.upper()}"
                
        ''' Tokenise Sentence & add tags to not tagged words (O)'''
        
        # 2. add token tag to main tag dictionary

        tags = []
        sentence = re.sub(self.PUNCTUATION_PATTERN, r" \1 ", sentence)
        for w in sentence.split():
            if w not in word_to_tag:
                tags.append("O")
            else:
                tags.append(word_to_tag[w])
                self.__add_tag(word_to_tag[w])
        
        return tags
    
    '''
    
    TAG CONVERSION
    
    '''
    # to word2id (tag_to_id)
    # to id2word (id_to_tag)

    def __add_tag(self, tag: str):
        if tag in self.tag_to_id:
            return
        id_ = len(self.tag_to_id)
        self.tag_to_id[tag] = id_
        self.id_to_tag[id_] = tag
        
    ''' Get Tag Number ID '''
    # or just number id for token
        
    def get_id(self, tag: str):
        return self.tag_to_id[tag]
    
    ''' Get Tag Token from Number ID'''
    # given id get its token
    
    def get_label(self, id_: int):
        return self.get_tag_label(id_)
    
parser = Parser()
parser(train_dataset["utt"][0], train_dataset["annot_utt"][0])
