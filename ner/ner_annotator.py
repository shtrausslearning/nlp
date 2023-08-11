#!/usr/bin/env python3

#from IPython.display import clear_output  # use if in jupyter
import pandas as pd
import numpy as np
import warnings;warnings.filterwarnings('ignore')

# NER Annotation Class

class ner_annotator:
	
	def __init__(self,df:pd.DataFrame):
		self.df = df
		self.word2tag = {}
		self.LABEL_PATTERN = r"\[(.*?)\]"
		self.deactive_df = None
		self.active_df = None
		
		self.__initialise()
		
	def __initialise(self):
		
		'''
		
		[1] ANNOTATION COLUMN RELATED OPERATIONS
		
		'''
		
		# if annotaion column is all empty
		
		if('annotated' in self.df.columns):
			
			if(self.df['annotated'].isna().sum() == self.df.shape[0]):
				self.df['annotated'] = None
				
			# if annotation column is not empty
				
			elif(self.df['annotated'].isna().sum() != self.df.shape[0]):
				
				# Store Tags
				for idx,row_data in self.df.iterrows():
					if(type(row_data['annotated']) == str):
						matches = re.findall(self.LABEL_PATTERN, row_data['annotated'] )
						for match in matches:
							tag, phrase = match.split(" : ")
							self.word2tag[phrase] = tag
							
		# if annotation column is not present
							
		else:
			word2tag = {}
			self.df['annotated'] = None    
			
		# active_df -> NaN are present
		# deactive_df -> already has annotations
			
		self.active_df = self.df[self.df['annotated'].isna()]
		self.deactive_df = self.df[~self.df['annotated'].isna()]
		
	'''
	
	REVIEW ANNOTATIONS
	
	'''
	# nullify rows which are not NaN, but don't have 
		
	def review_annotations(self):
		idx = list(self.deactive_df[~self.deactive_df["annotated"].str.contains(self.LABEL_PATTERN)]['annotated'].index)
		annot = list(self.deactive_df[~self.deactive_df["annotated"].str.contains(self.LABEL_PATTERN)]['annotated'].values)
		
		for i,j in zip(idx,annot):
			print(i,j)
			
	# drop annotations (from deactive_df)
			
	def drop_annotations(self,idx:list):
		remove_df = self.deactive_df.iloc[idx]
		remove_df['annotated'] = None
		self.active_df = pd.concat([self.active_df,remove_df])
		self.deactive_df = self.deactive_df.drop(list(idx),axis=0)
		self.deactive_df.sort_index()
		print('dopped annotations saving >> annot.csv')
		pd.to_csv('annot.csv',pd.concat([self.active_df,self.deactive_df]))
		
	'''
	
	ANNOTATE ON ACTIVE ONLY
	
	'''
		
	def ner_annotate(self):
		
		for idx,row_data in self.active_df.iterrows():
			
			q = row_data['question'] # question
			t = q                    # annotated question holder
			
			annotate_row = True
			while annotate_row is True:
				
				print('Current Annotations:')
				print(t,'\n')
				
				# user input
				user = input('tag (word-tag) format >> ')
				
				# [1] end of annotation (go to next row)
				
				if(user in ['quit','q']):
					
					annotate_row = False
					row_data['annotated'] = t
					
					# Store Tags
					matches = re.findall(self.LABEL_PATTERN, t)
					for match in matches:
						tag, phrase = match.split(" : ")
						self.word2tag[phrase] = tag
						
						# clean up output
						#               clear_output(wait=True)
						
						# [2] stop annotation loop
						
				elif(user in 'stop'):
					
					ldf = pd.concat([self.deactive_df,self.active_df],axis=0)
					ldf.to_csv('annot.csv',index=False)
					return 
				
				# [3] Reset current Row Tags
				
				elif(user in ['reset','r']):
					
					t=q 
					print(t,'\n')
					#           clear_output(wait=True)
					user = input('tag (word-tag) format >> ')
					
					# [4] Show current 
					
				elif(user == 'show'):
					print(self.word2tag)
					
				elif(user == 'dict'):
					
					# use dictionary to automatically set tags
					for word,tag in self.word2tag.items():
						if(word in t):
							express = f'[{tag} : {word}]' 
							t = t.replace(word,express)            
							
							# Tags Specified
							
				elif('-' in user):
					
					# parse input
					word,tag = user.split('-')
					
					if(word == ''):
						word = input('please add word >> ')
					if(tag == ''):
						tag = input('please add tag >> ')
						
					if(word in t):
						express = f'[{tag} : {word}]' 
						t = t.replace(word,express)
					else:
						print('not found in sentence')
						
				else:
					print('please use (word-tag format)')
					
					
					
df_annot = pd.read_csv('annot.csv')   # read dataframe
temp = ner_annotator(df_annot)   # start annotating documents
#temp.drop_annotations([3,4])   # drop annotations 
temp.review_annotations()      # show annotated rows
#print(temp.word2tag)          
#temp.ner_annotate()

