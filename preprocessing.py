import pandas as pd
import re
from pandas import DataFrame
from bs4 import BeautifulSoup

class Preprocessor:
    """
        Args:
            data: a pandas Dataframe object.
        
        remove_html_tags: uses bs4 to clean HTML tags.
            Args:
                self
                text: a String containing HTML tags.

        remove_pattern_from_column: removes a certain pattern from a column.
            Args:
                self
                column: specifies the column. Expects <str> or <int>.
                pattern: specifies the pattern. Expects an r-String.

        appy_method_to_column: applies a custom method to a column.
            Args:
                self
                column: specifies the column. Expects <str> or <int>.
                method: takes in a method that should be applied to the column. 
    """

    def __init__(self, data:DataFrame):
        self.data = data

    # ALL METHODS

    # CUSTOM METHODS FOR APPLY_METHOD_TO_COLUMN
    def remove_html_tags(self,text):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        return cleaned_text
    
    # COLUMN SPECIFIC METHODS

    def remove_pattern_from_column(self, column, pattern):
        try:
            re.compile(pattern)
            regex = True
        except re.error:
            print("Non valid regex pattern")
            regex = False
        
        self.data[column] = self.data[column].str.replace(pattern, '', regex=regex)
        
        print(f"removed {pattern} from column {column}.")
        return self.data
    
    def apply_method_to_column(self, column, method):
        self.data[column] = self.data[column].apply(method)
        return self.data
    
    





