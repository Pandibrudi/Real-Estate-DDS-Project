import pandas as pd
import re
from pandas import DataFrame
from bs4 import BeautifulSoup

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return str(cleaned_text)