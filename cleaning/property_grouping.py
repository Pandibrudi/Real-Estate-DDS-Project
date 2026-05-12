import pandas as pd
"""
Generally needed just incase we do more marketing analysis, rather than pricing -- as they are so minor in granualirty i made a map of broader areas
"""
property_type_keywords = {
    "detached house": "Detached",
    "semi-detached house": "Semi-Detached",
    "terraced house": "Terraced",
    "end of terrace": "End of Terrace",
    "town house": "Town House",
    "mews house": "Mews",
    "apartment": "Apartment",
    "flat": "Flat",
    "maisonette": "Maisonette",
    "penthouse": "Penthouse",
    "duplex": "Duplex",
    "villa": "Villa",
    "house": "House",  
}
property_group_map = {
    "Flat": "Flat",
    "Apartment": "Flat",
    "Ground Flat": "Flat",
    "Maisonette": "Flat",
    "Duplex": "Flat",
    "Penthouse": "Penthouse",
    "Detached": "Detached House",
    "Link Detached House": "Detached House",
    "Villa": "Detached House",
    "Semi-Detached": "Semi-Detached House",
    "Terraced": "Terraced House",
    "End of Terrace": "Terraced House",
    "Mews": "Terraced House",
    "Character Property": "House",
    "House": "House",
    "Town House": "Town House",
    "Land": "Land",
    "Plot": "Land",
    "Equestrian Facility": "Land",
    "Block of Apartments": "Other",
    "Not Specified": None,
}

def extract_property_type_from_description(description):
    if pd.isna(description):
        return None
    text = str(description).lower()
    for keyword, property_type in property_type_keywords.items():
        if keyword in text:
            return property_type
    return None

def extract_property_type_from_title(title):
    if pd.isna(title):
        return None
    title_lower = str(title).strip().lower()
    for keyword, property_type in property_type_keywords.items():
        if keyword in title_lower:
            return property_type
    return None

def clean_property_type(df, column="propertyType"):
    df["property_group"] = df[column].map(property_group_map)
    mask = df["property_group"].isna()
    df.loc[mask, column] = df.loc[mask, "title"].apply(extract_property_type_from_title)
    df.loc[mask, "property_group"] = df.loc[mask, column].map(property_group_map)
    
    mask = df["property_group"].isna()
    df.loc[mask, column] = df.loc[mask, "descriptionHtml"].apply(extract_property_type_from_description)
    df.loc[mask, "property_group"] = df.loc[mask, column].map(property_group_map)
    
    
    return df

def fix_property_group_outliers(df):
    df.loc[(df["property_group"] == "Flat") & (df["bedrooms"] >= 10), "property_group"] = "Other"
    return df