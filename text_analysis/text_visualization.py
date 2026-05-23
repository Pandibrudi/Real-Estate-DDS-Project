import os
import re
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ngram_visuals(data:pd.DataFrame):
    data = data.copy()
    data["ngram"] = data["ngram"].apply(lambda x: tuple(map(str, x)))
    data["ngram"] = data["ngram"].apply(lambda x: " ".join(x))
    plt.figure(figsize=(10, 6))

    sns.barplot(data=data, x="frequency", y="ngram", color="steelblue")

    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.title("n-gram Frequencies")

    sns.despine(left=True, bottom=True)

    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M", t)
    output_file = f"text_analysis/visuals/n_grams-{timestamp}.png" # relative path only works when executed through main, needs to be created in case it's missing

    plt.savefig(output_file, bbox_inches="tight")
    plt.show()

    return output_file