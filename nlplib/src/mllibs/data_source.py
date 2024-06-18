
'''
####################################################################################

                             Data Sources Class

####################################################################################
'''

class data_storage:
    
    def __init__(self):
        self.data = {}
        
    def add_pdf(self,data:pd.DataFrame,data_name:str):
        self.data[data_name] = data
        
    def get_pdf_colnames(self,data_name:str):
        return self.data[data_name].columns.tolist()
        
    def show_data(self):
        print(self.data)
        
    def show_data_names(self):
        return list(self.data.keys())
