
import numpy as np
from bs4 import BeautifulSoup
import requests
import nltk
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
from time import sleep 

# class to store parsed wikipedia page data

class dbWiki:
	
	def __init__(self):
		 
		self.title = None            # title of parsed page
		self.text = []               # store paragraphs
		self.sentences = []          # store sentences
		self.pidx = []               # sentence paragraph index
		self.topic = None            # got topic
	
	# scrape wikipedia article
	def scrape_wiki(self,topic):
		
		# process topic as required by Wikipedia URL system
		topic = topic.lower().strip().capitalize().split(' ')
		topic = '_'.join(topic)
		
		# Try to receive the topic
		
		try:

			# parse website
			link = 'https://en.wikipedia.org/wiki/'+ topic # creata an url
			data = requests.get(link).content              # access contents via url
			soup = BeautifulSoup(data, 'html.parser')      # parse data as soup object
			
			# find tag contents 
			_p = soup.findAll('p')  # scrape html tag 'p'
			_dd = soup.findAll('dd') # scrape html tag 'dd'
			_li = soup.findAll('li') # scrape html tag 'li'
			
			# store tags
			ptags = [p for p in _p]
			ddtags = [dd for dd in _dd]
			#litags = [li for li in _li]
			alltags = ptags + ddtags
			
			# iterate over all data
			for tag in alltags: 

				temp = []
				# iterate over para, desc data 
				for content in tag.contents:
					
					if(content.name != 'sup' and content.string != None):
						stripped = ' '.join(content.string.strip().split())
						temp.append(stripped)
						
				# make into a single string
				self.text.append(' '.join(temp))
			
			# obtain sentences from paragraphs
			for i,para in enumerate(self.text):
				
				_sent = nltk.sent_tokenize(para) # sentence tokenise
				self.sentences.extend(_sent)     # store tokenised sentences
				
				# useful in case user prompts "more" info
				index = [i]*len(_sent)   # map sentence to paragraph index
				self.pidx.extend(index)
				
			self.title = soup.find('h1').string  # extract header
			self.topic = True                                  # turn respective flag on
			print('wiki >> ',self.title,' received!') 
			
		# in case of unavailable topics
		except Exception as e:
			print('>>  Error: {}. \
			Please input some other topic!'.format(e))

get_wiki = dbWiki()
get_wiki.scrape_wiki('Emissivity')
# wiki >>  Emissivity  received!

get_wiki.sentences[0]
# 'The emissivity of the surface of a material is its effectiveness in emitting energy as thermal radiation .'
