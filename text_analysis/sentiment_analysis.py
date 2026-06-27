from tqdm import tqdm
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

def sentiment_analysis(df:pd.DataFrame, column:str, granularity="doc"):
    # still thinking about using granularity to get word level sentiment    
    def get_sentiment(text, granularity=granularity):
        doc = nlp(text)
        polarity = doc._.blob.polarity
        return polarity
    
    new_df = df.copy()
    tqdm.pandas(desc=f"Getting sentiment for column '{column}'")
    new_df[f"{column}_sentiment_score"] = new_df[column].progress_apply(get_sentiment)
    return new_df

def label_sentiment(df:pd.DataFrame, column:str, threshold:float=0.26):
    # adjust threshold according to distribution!
    labels = ["Neutral", "Positive", "Negative"]
    def determine_label(score:float, threshold=threshold):
        if score >= threshold:
            return labels[1]
        elif score < threshold:
            return labels[2]
        else:
            return labels[0]
    
    new_df = df.copy()
    tqdm.pandas(desc=f"Labelling sentiments for column '{column}'")
    new_df[f"{column}_sentiment"] = new_df[column].progress_apply(determine_label)
    return new_df

def get_sentiment_distribution(df:pd.DataFrame, column:str):
    dist = {}
    dist["q25"] = df[column].quantile(0.25)
    dist["q75"] = df[column].quantile(0.75)
    dist["iqr"] = dist["q75"] - dist["q25"]    

    # console printout - could be commented out
    print(f"25th percentile: {dist['q25']:.4f}")
    print(f"75th percentile: {dist['q75']:.4f}")
    print(f"IQR: {dist['iqr']:.4f}")
    print(df[column].describe())
    return df, dist


