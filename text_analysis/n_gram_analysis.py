import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

def get_ngrams(df: pd.DataFrame, column: str, n: int, id_col: str = "id"):
    lengths = df[column].str.len().to_numpy()

    flat = np.concatenate(df[column].to_numpy())
    doc_ids = np.repeat(df[id_col].to_numpy(), lengths)

    windows = sliding_window_view(flat, n)
    window_docs = sliding_window_view(doc_ids, n)

    valid_mask = np.zeros(len(windows), dtype=bool)

    start = 0
    pbar = tqdm(lengths, desc=f"Getting n-grams (n={n}) for '{column}'")

    for length in pbar:
        if length >= n:
            valid_mask[start:start + length - n + 1] = True
        start += length

    ngrams = windows[valid_mask]
    docs = window_docs[:, 0][valid_mask]  # all tokens in a window share same doc_id

    new_df = pd.DataFrame({"doc_id": docs, "ngram": list(map(tuple, ngrams))})
    print(new_df)
    return new_df

def get_ngram_frequencies(df:pd.DataFrame, top_k:int=50):
    pbar = tqdm(df["ngram"], desc="Getting n-gram frequencies")
    ngram_tuples = [tuple(row) for row in pbar]
    frequencies = Counter(ngram_tuples)
    top_ngrams = frequencies.most_common(top_k)
    top_ngrams = pd.DataFrame(top_ngrams, columns=["ngram", "frequency"])
    return top_ngrams