import re
import spacy
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter
from nltk import ngrams


nlp = spacy.load("en_core_web_sm")


def clean_string(text:str):
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_column(df:pd.DataFrame, column:str, remove_stopwords:bool = True, lowercase:bool = True):
    def tokenize(text):
        doc = nlp(str(text))
        tokens = []
        for token in doc:
            if not token.is_space and not token.is_punct:
                if not (remove_stopwords and token.is_stop):
                    tok = token.text.lower() if lowercase else token.text
                    tokens.append(tok)
        return tokens

    new_df = df.copy()
    tqdm.pandas(desc=f"Tokenizing column '{column}'")
    new_df[f"{column}_tokens"] = new_df[column].progress_apply(tokenize)
    
    return new_df



