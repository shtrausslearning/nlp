import pandas as pd
from mllibs.data_source import data_storage
from mllibs.user_request import user_request


'''
####################################################################################

                             Main Assembly Class 

####################################################################################
'''

class nlpi:

    def __init__(self):
        self.nlpi_data = data_storage()
        self.request = user_request(self.nlpi_data)

    # evaluate user query
    def query(self,request:str):
        self.request.store_tokens(request)
        self.request.evaluate()
        
    # add data to data sources 
    def add(self,data,name:str):
        if(isinstance(data,pd.DataFrame)):
            self.nlpi_data.add_pdf(data,name)
