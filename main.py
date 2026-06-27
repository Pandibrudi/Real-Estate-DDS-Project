import os
import uuid

import pandas as pd

from data.text_constants import BOILERPLATE, STOP_WORDS
from dataloader import load_data
from cleaning.date_clean import process_added_on_column
from cleaning.price_clean import price_gbp
from cleaning.property_grouping import clean_property_type, fix_property_group_outliers
from cleaning.sqr_feet_clean import clean_size
from cleaning.bedroom_and_bathroom_clean import clean_bedrooms_and_bathrooms
from cleaning.location_extraction import clean_location
from cleaning.school_features import enrich_data
from text_analysis.text_preprocessing import clean_string, tokenize_column, add_stop_words, remove_boilerplate
from text_analysis.n_gram_analysis import get_ngrams, get_ngram_frequencies, ngrams_per_quartile
from text_analysis.text_visualization import ngram_visuals, sent_dist_visuals, ngram_quartile_visuals
from text_analysis.sentiment_analysis import sentiment_analysis, label_sentiment, get_sentiment_distribution


def clean(df):
    df["price"] = df["price"].apply(price_gbp)
    date_df = process_added_on_column(df, column="addedOn")
    df = pd.concat([df, date_df], axis=1)
    df = clean_property_type(df)
    df = clean_size(df)
    df = clean_bedrooms_and_bathrooms(df)
    df = fix_property_group_outliers(df)
    df = clean_location(df)
    # IMPORTANT! 
    # only enrich properties with school data after index is set.
    df = df.drop(columns=["category", "size_missing", "addedOn"])
    df.to_csv("data/cleaned_data.csv", index=False)
    return df


def text_analysis(df, text_column="descriptionHtml", n: int = 2, q_top_k: int = 15,
                   freq_top_k: int = 30, stop_words: list = None, boilerplate: list = None):
    stop_words = stop_words or []
    boilerplate = boilerplate or []

    # cleaning
    df[text_column] = df[text_column].apply(clean_string)
    df[text_column] = df[text_column].apply(lambda x: remove_boilerplate(x, boilerplate))

    # sentiment analysis
    df = sentiment_analysis(df, text_column)
    sen_dist = get_sentiment_distribution(df, f"{text_column}_sentiment_score")
    sent_dist_visuals(sen_dist[0], sen_dist[1], column=f"{text_column}_sentiment_score")
    df = label_sentiment(df, f"{text_column}_sentiment_score")

    # register stop words before tokenizing so they're filtered out downstream
    add_stop_words(stop_words)

    df = tokenize_column(df, text_column)
    token_column = f"{text_column}_tokens"  # output column name from tokenize_column

    ngrams = get_ngrams(df, token_column, n=n)
    quartile_counts = ngrams_per_quartile(df, ngrams, feature="price", top_k=q_top_k)
    ngram_quartile_visuals(quartile_counts, feature="price", top_k=q_top_k)
    ngram_freq = get_ngram_frequencies(ngrams, top_k=freq_top_k)
    ngram_visuals(ngram_freq)
    return df


PATH = "data/realestate_data_london_2024_nov.csv"
CLEANED_PATH = "data/cleaned_data.csv"


def main():
    if os.path.exists(CLEANED_PATH):
        df = load_data(CLEANED_PATH)
    else:
        df = load_data(PATH)
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df = clean(df)

    df = enrich_data(df)
    df = text_analysis(df, stop_words=STOP_WORDS, boilerplate=BOILERPLATE)
    return df


if __name__ == "__main__":
    df = main()