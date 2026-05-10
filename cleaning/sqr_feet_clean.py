
import pandas as pd 
import re

"""
If Land we won't have size in sqfeet
We also sometimes have this in the description, under the assumption we are talking about the interior sqr feet
"""


def extract_size_from_description(description):
    if pd.isna(description):
        return None
    text = str(description).lower()
    
  
    property_indicators = [
    r"(?:spanning|spans|extends? to|totalling|totaling|measures?|comprises?|approximately|approx\.?)\s+(?:an?\s+)?(?:impressive\s+|expansive\s+)?(\d[,\d]*\.?\d*)\s*(?:square\s+feet|sq\.?\s*ft\.?|sq\.|sqft)",
    r"(\d[,\d]*\.?\d*)\s*(?:square\s+feet|sq\.?\s*ft\.?|sq\.|sqft)\s+(?:of|interior|living)",
    r"(\d[,\d]*\.?\d*)\s*(?:sq\.?\s*m\.?|sqm|m2)\s+(?:of|interior|living|gross)",
]
    
    for pattern in property_indicators:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1).replace(",", ""))
            if any(unit in match.group(0) for unit in ["sq m", "sqm", "m2"]):
                value *= 10.764
            return value
    
    return None

def extract_acres_from_description(description):
    if pd.isna(description):
        return None
    text = str(description).lower()
    match = re.search(r"(\d+\.?\d*)\s*acres?", text)
    return float(match.group(1))  * 43560 if match else None

def clean_size(df, column="sizeSqFeetMax", group_column="property_group"):
    df["size_missing"] = df[column].isna()

    missing_mask = df[column].isna()
    df.loc[missing_mask, column] = df.loc[missing_mask, "descriptionHtml"].apply(extract_size_from_description)


    land_mask = df[group_column] == "Land"
    df.loc[land_mask & df[column].isna(), column] = df.loc[land_mask & df[column].isna(), "descriptionHtml"].apply(extract_acres_from_description)

    non_land_mask = ~land_mask & df[column].isna()
    median_by_group = df.groupby(group_column)[column].transform("median")
    df.loc[non_land_mask, column] = median_by_group[non_land_mask]
    return df