import pandas as pd 
import re

word_to_num = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def extract_bedrooms_from_title(title):
    if pd.isna(title):
        return None
    match = re.search(r"(\d+)\s+bedroom", str(title).lower())
    return int(match.group(1)) if match else None

def extract_bedrooms_from_description(description):
    if pd.isna(description):
        return None
    text = str(description).lower()
    match = re.search(r"(\d+)\s+bedroom", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(one|two|three|four|five|six|seven|eight|nine|ten)(?:/(one|two|three|four|five|six|seven|eight|nine|ten))?\s+bedroom", text)
    if match:
        return word_to_num[match.group(1)] 
    return None


def clean_bedrooms_and_bathrooms(df, group_column = "property_group" ):
    land_mask = df[group_column] == "Land"
    mask = df["bedrooms"].isna()
    df.loc[mask, "bedrooms"] = df.loc[mask, "title"].apply(extract_bedrooms_from_title)
    mask = df["bedrooms"].isna()
    df.loc[mask, "bedrooms"] = df.loc[mask, "descriptionHtml"].apply(extract_bedrooms_from_description)
    df.loc[land_mask, "bedrooms"] = df.loc[land_mask, "bedrooms"].fillna(0)
    df.loc[land_mask, "bathrooms"] = df.loc[land_mask, "bathrooms"].fillna(0)
    non_land_mask_bed = ~land_mask & df["bedrooms"].isna()
    non_land_mask_bath = ~land_mask & df["bathrooms"].isna()
    median_bathroom_group = df.groupby(group_column)["bathrooms"].transform("median")
    median_bedroom_group = df.groupby(group_column)["bedrooms"].transform("median")
    df.loc[non_land_mask_bed,"bedrooms"] = median_bathroom_group[non_land_mask_bed]
    df.loc[non_land_mask_bath,"bathrooms"] = median_bedroom_group[non_land_mask_bath]

    return df

