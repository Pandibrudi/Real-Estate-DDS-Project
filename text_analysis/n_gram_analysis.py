import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

def get_ngrams(df:pd.DataFrame, column:str, n:int):

    lengths = df[column].str.len().to_numpy()

    flat = np.concatenate(df[column].to_numpy())

    windows = sliding_window_view(flat, n)

    valid_mask = np.zeros(len(windows), dtype=bool)

    start = 0
    pbar = tqdm(lengths, desc=f"Getting n-grams (n={n}) for '{column}'")
    for length in pbar:
        if length >= n:
            valid_mask[start:start + length - n + 1] = True
        start += length

    return windows[valid_mask]

def get_ngram_frequencies(ngrams:pd.DataFrame, top_k=50):
    pbar = tqdm(ngrams, desc="Getting n-gram frequencies")
    ngram_tuples = [tuple(row) for row in pbar]
    frequencies = Counter(ngram_tuples)
    top_ngrams = frequencies.most_common(top_k)
    top_ngrams = pd.DataFrame(top_ngrams, columns=["ngram", "frequency"])
    return top_ngrams