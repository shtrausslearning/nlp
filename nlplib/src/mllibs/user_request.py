
import pandas as pd
import re

'''
####################################################################################

                             User Request Class

####################################################################################
'''


class user_request:
    
    def __init__(self,data):
        self.nlpi_data = data                # data class instance

    def store_tokens(self,request:str):
        self.tokens = self.tokenise(request) # tokenised request

    @staticmethod
    def tokenise(text:str) -> list:
        pattern = r'\w+'
        words = re.findall(pattern, text)
        return words

    def evaluate(self):
        self.data_in_tokens = [True if i in self.nlpi_data.show_data_names() else False for i in self.tokens] 
        self.check_tokens_for_pdf_columns() # self.column_in_tokens
        self.find_neighbouring_tokens()  # self.merged_tokens
        self.column_name_groupings()   # self.grouped_col_names

    def check_tokens_for_pdf_columns(self):
        
        '''
        
        Check if request token is in request data token column names
        
        '''
        
        # initialise
        self.column_in_tokens = [None] * len(self.tokens)
        
        # loop through data tokens
        for data_token,token in zip(self.data_in_tokens,self.tokens):
            if(data_token):
                data_columns = self.nlpi_data.get_pdf_colnames(token)
                for ii,ltoken in enumerate(self.tokens):
                    if(ltoken in data_columns):
                        self.column_in_tokens[ii] = token    
                        
    def find_neighbouring_tokens(self):
        
        '''
        
        group together tokens which are connected by [and], [,]
        
        '''

        comma_indices = [i for i, token in enumerate(self.tokens) if token == ',']
        and_indices = [i for i, token in enumerate(self.tokens) if token == 'and']
        merging_tokens = [',','and']

        tuples_list = []
        for ii,token in enumerate(self.tokens):
            if(token in merging_tokens):
                tuples_list.append([self.tokens[ii-1],self.tokens[ii+1]])   
            
        if(len(tuples_list) > 1):
            
            merged_tuples = None
            for ii in range(len(tuples_list) - 1):
                tlist = tuples_list[ii].copy()
                tlist.extend(tuples_list[ii+1])
                merged_tuples = set(tlist)

            self.merged_tokens = merged_tuples
            
        else:
            self.merged_tokens = set(tuples_list[0])
        
    def column_name_groupings(self):
        
        '''
            
        Group together data column names
        
        '''
            
        all_idx = []       
        indices = [index for index, value in enumerate(self.column_in_tokens) if value is not None]

        def remove_elements(lst):
            i = 1
            while i < len(lst) - 1:
                if (i == 1 and abs(lst[i] - lst[i+1]) != 2) and (i == len(lst) - 2 and abs(lst[i] - lst[i-1]) != 2) and (abs(lst[i] - lst[i-1]) != 2 and abs(lst[i] - lst[i+1]) != 2):
                    del lst[i]
                else:
                    i += 1

            # Check the first element with its right neighbour
            try:
                if abs(lst[0] - lst[1]) != 2:
                    del lst[0]
            except:
                pass
            
            try:
                # Check the last element with its left neighbour
                if abs(lst[-1] - lst[-2]) != 2:
                    del lst[-1]
            except:
                pass
                    
            return lst
        
        indices = remove_elements(indices)
                
        # all column name tokens
        for ii,token in enumerate(self.column_in_tokens):
            if(token is not None):
                all_idx.append(ii)   
                
        def find_nested_combination(main_list:list,pattern_list:list) -> list:
            for i in range(len(main_list) - len(pattern_list) + 1):
                if main_list[i:i+len(pattern_list)] == pattern_list:
                    main_list[i:i+len(pattern_list)] = [pattern_list]
            return main_list
        
        if(len(indices) > 1):
            self.grouped_col_names = find_nested_combination(all_idx,indices)
        else:
            self.grouped_col_names = all_idx
        
    def token_info(self):
        return pd.DataFrame({'token':self.tokens,
                             'data_id':self.data_in_tokens,
                             'col_id':self.column_in_tokens})
        
