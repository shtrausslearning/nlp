#!/usr/bin/env python3

import spacy
import re
from spacy import displacy

# manual spacy NER label visualisation

def make_spacy_ents(text,words_list,tag_list):
	map_dict = dict(zip(words_list,tag_list))
	
	def find_idx(text,words):
		
		lst_output = []
		for word in words:
			for m in re.finditer(word,text):
				lst_output.append({'start':m.start(),'end':m.end(),'label':map_dict[m.group()]})
				
		return lst_output
	
	lst_ents = find_idx(text,words_list)
	print(lst_ents)

	# make spacy ents
	dic_ents = {"text": text,"ents": lst_ents,"title": None}
	displacy.serve(dic_ents,manual=True, style="ent")


text= "My name is John Smith and I live in Paris"
words_list= ["John Smith",'Paris']
tag_list= ['person','location']

make_spacy_ents(text,words_list,tag_list)
