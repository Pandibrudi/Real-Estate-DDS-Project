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
    plt.figure(figsize=(20, 12))

    sns.barplot(data=data, x="frequency", y="ngram", color="steelblue")

    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.title("n-gram Frequencies")

    sns.despine(left=True, bottom=True)

    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M", t)
    # when you run this the first time, please create a "visuals" folder in "text_analysis"
    output_file = f"text_analysis/visuals/n_grams-{timestamp}.png"

    plt.savefig(output_file, bbox_inches="tight")
    plt.show()

    return output_file


def sent_dist_visuals(df, dist:dict, column:str, save:bool=True):

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[column], kde=True, bins=30, ax=ax, color='steelblue')

    ax.axvline(dist["q25"], color='orange', linestyle='--', label=f'P25 ({dist["q25"]:.2f})')
    ax.axvline(dist["q75"], color='red',    linestyle='--', label=f'P75 ({dist["q75"]:.2f})')
    ax.legend()

    if save:
        t = time.localtime()
        timestamp = time.strftime("%b-%d-%Y_%H%M", t)
        # when you run this the first time, please create a "visuals" folder in "text_analysis"
        output_file = f"text_analysis/visuals/sentiment_dist-{timestamp}.png"
        plt.savefig(output_file)

    plt.show()
    return dist