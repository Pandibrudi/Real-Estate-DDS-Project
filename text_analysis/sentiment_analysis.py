from tqdm import tqdm
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

def sentiment_analysis(df:pd.DataFrame, column:str):
    def get_sentiment(text):
        doc = nlp(text)
        polarity = doc._.blob.polarity  
        
        return polarity
    
    new_df = df.copy()
    tqdm.pandas(desc=f"Getting sentiment for column '{column}'")
    new_df[f"{column}_sentiment_score"] = new_df[column].progress_apply(get_sentiment)
    new_df.to_csv("test.csv")
    return new_df
