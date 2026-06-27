import ast
import pandas as pd
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

def get_ngrams(df: pd.DataFrame, column:str, n:int, id_col:str="id"):
    records = []
    pbar = tqdm(df.iterrows(), desc=f"Getting n-grams for {column}")
    for _, row in pbar:
        text = row[column]
        if text is not None:
            tokens = text
            for gram in ngrams(tokens, n):
                records.append({"id": row[id_col], "ngram": gram})
    new_df = pd.DataFrame(records, columns=["id", "ngram"])
    return new_df

def get_ngram_frequencies(df:pd.DataFrame, top_k:int=50):
    pbar = tqdm(df["ngram"], desc="Getting n-gram frequencies")
    ngram_tuples = [tuple(row) for row in pbar]
    frequencies = Counter(ngram_tuples)
    top_ngrams = frequencies.most_common(top_k)
    top_ngrams = pd.DataFrame(top_ngrams, columns=["ngram", "frequency"])
    return top_ngrams

def ngrams_per_quartile(df:pd.DataFrame, ngram_df:pd.DataFrame, feature:str, top_k:int = 10):
    ngram_df['ngram'] = ngram_df['ngram'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    result = ngram_df.groupby('id')['ngram'].apply(list).reset_index()
    merged = pd.merge(df, result, on="id")
    merged['quartile'] = pd.qcut(merged['price'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    quartile_ngram_counts = {}

    for name, group in merged.groupby('quartile'):
        all_ngrams = [ngram for ngrams in group['ngram'] for ngram in ngrams]
        counts = Counter(all_ngrams)
        counts_formatted = {str(k): v for k, v in counts.items()}
        quartile_ngram_counts[name] = (pd.DataFrame(counts_formatted.items(), columns=['ngram', 'count']).sort_values('count', ascending=False).reset_index(drop=True))
        low = int(group[feature].min())
        high = int(group[feature].max())
        quartile_ngram_counts[name].attrs['range'] = (low, high)

    return quartile_ngram_counts






