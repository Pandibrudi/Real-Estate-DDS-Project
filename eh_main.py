import pandas as pd
from dataloader import load_data
from cleaning.date_clean import process_added_on_column
from cleaning.price_clean import price_gbp
from cleaning.property_grouping import clean_property_type, fix_property_group_outliers
from cleaning.sqr_feet_clean import clean_size
from cleaning.bedroom_and_bathroom_clean import clean_bedrooms_and_bathrooms
from cleaning.location_extraction import clean_location, enrich_postcodes
import uuid
import duckdb

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
    #used for cleaning anchors - we drop them as they are used for only analysis.
    df = df.drop(columns=["category", "size_missing", "addedOn"])
    df.to_csv("data/cleaned_data.csv", index=False)
    return df

def main(path):
    df = load_data(path)
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    #after we run this once we have new csv file, so you can then just use 
    #df = load_data('data/cleaned_data.csv') - this should reduce the unnecessary processing if you want to explore the data
    df = clean(df)
    return df
PATH="data/realestate_data_london_2024_nov.csv"

if __name__ == "__main__":
    df = main(PATH)
#test on the columns with good old duck db
print(duckdb.query("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(price) as has_price,
        COUNT(date_field) as has_date,
        COUNT(propertyType) as has_property_type,
        COUNT(property_group) as has_property_group,
        COUNT(sizeSqFeetMax) as has_size,
        COUNT(bedrooms) as has_bedrooms,
        COUNT(bathrooms) as has_bathrooms,
        COUNT(street) as has_street,
        COUNT(area) as has_area,
        COUNT(postcode) as has_postcode,
        COUNT(latitude) as has_latitude,
        COUNT(district) as has_district,
        COUNT(ward) as has_ward,
        COUNT(constituency) as has_constituency
    FROM df
""").to_df())


