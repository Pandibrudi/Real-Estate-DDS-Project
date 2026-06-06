import os

import pandas as pd
from dataloader import load_data
from cleaning.date_clean import process_added_on_column
from cleaning.price_clean import price_gbp
from cleaning.property_grouping import clean_property_type, fix_property_group_outliers
from cleaning.sqr_feet_clean import clean_size
from cleaning.bedroom_and_bathroom_clean import clean_bedrooms_and_bathrooms
from cleaning.location_extraction import clean_location
from cleaning.school_features import enrich_data
import uuid


#after you run this once, creating the 
def clean (df):
    df["price"] = df["price"].apply(price_gbp)
    date_df = process_added_on_column(df, column="addedOn")
    df = pd.concat([df, date_df], axis=1)
    df = clean_property_type(df)
    df = clean_size(df)
    df = clean_bedrooms_and_bathrooms(df)
    df = fix_property_group_outliers(df)
    print('clean location')
    df = clean_location(df)
    #!! IMPORTANT
    #only enrich properties with school data after index set.
    df = df.drop(columns=["category", "size_missing", "addedOn"])
    df.to_csv("data/cleaned_data.csv", index=False)
    return df

def main(path):
         #after we run this once we have new csv file, so you can then just use 
     if path == 'data/cleaned_data.csv':
        df = load_data(path)
        df = enrich_data(df)
        return df
     else:
        df = load_data(path)
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df = clean(df)
        df = enrich_data(df)
        return df


PATH="data/realestate_data_london_2024_nov.csv"
if __name__ == "__main__":
    if not os.path.exists("data/cleaned_data.csv"):
        df = main(PATH)
    else :
        df = main('data/cleaned_data.csv')

