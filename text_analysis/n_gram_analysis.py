import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view
from nltk import ngrams

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

def get_ngrams(df: pd.DataFrame, column: str, n: int, id_col: str = "id") -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = row[column]
        if text is not None:
            tokens = text
            for gram in ngrams(tokens, n):
                records.append({"doc_id": row[id_col], "ngram": gram})
    new_df = pd.DataFrame(records, columns=["doc_id", "ngram"])
    return new_df

def get_ngram_frequencies(df:pd.DataFrame, top_k:int=50):
    pbar = tqdm(df["ngram"], desc="Getting n-gram frequencies")
    ngram_tuples = [tuple(row) for row in pbar]
    frequencies = Counter(ngram_tuples)
    top_ngrams = frequencies.most_common(top_k)
    top_ngrams = pd.DataFrame(top_ngrams, columns=["ngram", "frequency"])
    return top_ngrams