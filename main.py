import pandas as pd
from dataloader import load_data
from cleaning.date_clean import process_added_on_column
from cleaning.price_clean import price_gbp
from cleaning.property_grouping import clean_property_type, fix_property_group_outliers
from cleaning.sqr_feet_clean import clean_size
from cleaning.bedroom_and_bathroom_clean import clean_bedrooms_and_bathrooms
from cleaning.location_extraction import clean_location, enrich_postcodes
from cleaning.hmtl_clean import remove_html_tags

from text_analysis.text_preprocessing import clean_string, tokenize_column
from text_analysis.n_gram_analysis import get_ngrams, get_ngram_frequencies
from text_analysis.text_visualization import ngram_visuals
from text_analysis.sentiment_analysis import sentiment_analysis

import uuid
import duckdb

PATH="data/realestate_data_london_2024_nov.csv"

def clean (df):
    df["price"] = df["price"].apply(price_gbp)
    df["descriptionHtml"] = df["descriptionHtml"].apply(remove_html_tags)
    date_df = process_added_on_column(df, column="addedOn")
    df = pd.concat([df, date_df], axis=1)
    df = clean_property_type(df)
    df = clean_size(df)
    df = clean_bedrooms_and_bathrooms(df)
    df = fix_property_group_outliers(df)
    print('clean location')
    df = clean_location(df)
    #used for cleaning anchors - we drop them as they are used for only analysis.
    df = df.drop(columns=["category", "size_missing", "addedOn"])
    df.to_csv("data/cleaned_data.csv", index=False)
    return df

def text_analysis(df):
    df["descriptionHtml"] = df["descriptionHtml"].apply(clean_string)
    df = sentiment_analysis(df, "descriptionHtml")
    df = tokenize_column(df, "descriptionHtml")
    ngrams = get_ngrams(df, "descriptionHtml_tokens", 2) # descriptionHtml_tokens is the output column from the tokenizer function
    ngram_freq = get_ngram_frequencies(ngrams, top_k=20)
    ngram_visuals(ngram_freq)
    return df

def main(path):
    df = load_data(path)
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    #after we run this once we have new csv file, so you can then just use 
    df = load_data('data/cleaned_data.csv') 
    
    #df = clean(df) deactived while working on text_analysis
    ta = text_analysis(df)
    return df

if __name__ == "__main__":
    df = main(PATH)