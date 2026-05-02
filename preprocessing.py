import pandas as pd
from bs4 import BeautifulSoup

def remove_html_tags(data):
    column = data['descriptionHtml'].tolist()

    clean_data = [] # TODO should be doing this in the df directly, just testing
    for row in column:
        soup = BeautifulSoup(row, "html.parser")
        text = soup.get_text()
        clean_data.append(text)
    
    return clean_data
