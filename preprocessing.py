import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup

class Preprocessor:
    """
        Args:
            data: a pandas Dataframe object.
        
        remove_html_tags:
            Args:
                self
                text: a String containing html tags.
        
    """

    def __init__(self, data:DataFrame):
        self.data = data

    # ALL METHODS

    def remove_html_tags(self,text):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
            
        return cleaned_text
    
    def remove_double_whitespace(self, text):
        cleaned_text = text.replace("  ", " ")
        return cleaned_text
    
    # CLEANING METHODS

    def remove_pattern_from_column(self, column, pattern):
        self.data[column] = self.data[column].str.replace(pattern, '', regex=True)
        return f"removed {pattern} from column {column}."

    
    def multi_cleaning(self, column, methods:list):
        for m in methods:
            self.data[column] = self.data[column].apply(m)
        return self.data
    
    





