import os
import re
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def ngram_quartile_visuals(quartile_ngram_counts: dict, feature: str, top_k: int = 10, save: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(32, 20), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"Top {top_k} N-grams per {feature} Quartile", fontsize=22, fontweight='bold', y=1.02)
    palettes = {"Q1": "Greens_d", "Q2": "Blues_d", "Q3": "YlOrBr_r", "Q4": "Reds_r"}

    for ax, (quartile_name, data) in zip(axes, quartile_ngram_counts.items()):
        low, high = data.attrs['range']
        top_data = data.head(top_k)

        sns.barplot(data=top_data, x='count', y='ngram', ax=ax, hue='ngram', palette=palettes[quartile_name], legend=False)

        ax.set_title(f"{quartile_name}  |  {low:,} – {high:,} £", fontsize=18, fontweight='bold', pad=12)
        ax.set_xlabel("Count", fontsize=14)
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='x', labelsize=12)
        ax.bar_label(ax.containers[0], fontsize=11, padding=3)

    plt.tight_layout()

    if save:
        t = time.localtime()
        timestamp = time.strftime("%b-%d-%Y_%H%M", t)
        output_file = f"text_analysis/visuals/ngram_quartile_dist-{timestamp}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=150)

    plt.show()
    return quartile_ngram_counts