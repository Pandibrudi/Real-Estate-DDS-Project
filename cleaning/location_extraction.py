import os
import re
from duckdb import df
import pandas as pd
import requests
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2
skip_values = {"united kingdom", "england", "london", "greater london"}


def fetch_by_street_nominatim(street):
    if pd.isna(street):
        return {}
    if street:
        street = re.sub(r'^\d+\s+', '', street).strip()
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{street} London", "format": "json", "limit": 1},
            headers={"User-Agent": "london-realestate-analysis"},
            timeout=5
        )
        if response.status_code == 200:
            results = response.json()
            if results:
                r = results[0]
                postcode_match = re.search(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})\b', r["display_name"])
                postcode = postcode_match.group(0) if postcode_match else None
                return {
                    "latitude": float(r["lat"]),
                    "longitude": float(r["lon"]),
                    "postcode": postcode,
                }
    except:
        pass
    return {}

def fetch_postcode_data(postcode):
    if pd.isna(postcode):
        return {}
    
    query = f"{postcode}"
    
    try:
        response = requests.get(
            "https://api.postcodes.io/postcodes?q=" + query +"&limit=1",
            timeout=5
        )
        if response.status_code == 200:
        
            results = response.json().get("result", [])
            if results:
                r = results[0]
                return {
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "district": r["admin_district"],
                    "ward": r["admin_ward"],
                    "constituency": r["parliamentary_constituency"],
                }
    except:
        pass
    return {}

def load_postcode_cache(path, column="postcode"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.set_index('postcode').to_dict(orient="index")
    return {}

def resolve_postcode(postcode, cache):
    if postcode in cache:
        return cache[postcode]
    data = fetch_postcode_data(postcode)
    resolved = {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "district": data.get("district"),
        "ward": data.get("ward"),
        "constituency": data.get("constituency"),
    }
    cache[postcode] = resolved
    return resolved


def enrich_postcodes(df, cache_path="data/postcode_cache.csv", column="postcode"):

    cache = load_postcode_cache(cache_path, column)
    mask = df[column].notna()
    total = mask.sum()

    for i, (idx, row) in enumerate(df.loc[mask].iterrows()):
        postcode = row[column]
        if postcode in cache:
            print(f"[{i}/{total}] Cache hit: {postcode}")
            df.loc[idx, "latitude"] = cache[postcode].get("latitude")
            df.loc[idx, "longitude"] = cache[postcode].get("longitude")     
            continue

        data = resolve_postcode(postcode, cache)
        cache[postcode] = data

        for k, v in data.items():
            df.loc[idx, k] = v

        if i % 100 == 0:
            print(f"{i}/{total} processed")

    pd.DataFrame.from_dict(cache, orient="index") \
        .to_csv(cache_path, index_label="postcode")

    return df

def enrich_missing_from_street(df):

    missing = df["postcode"].isna() & df["street"].notna()

    for idx, row in df.loc[missing].iterrows():

        data = fetch_by_street_nominatim(row["street"])

        df.loc[idx, "latitude"] = data.get("latitude")
        df.loc[idx, "longitude"] = data.get("longitude")

        if data.get("postcode"):
            df.loc[idx, "postcode"] = data["postcode"]

    return df


def extract_location(title):
    if pd.isna(title):
        return {"street": None, "area": None, "postcode": None}

    match = re.search(r"for sale in (.+)", str(title), re.IGNORECASE)
    if not match:
        return {"street": None, "area": None, "postcode": None}
    parts = [p.strip() for p in match.group(1).split(",")]
    postcode_pattern = r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}|[A-Z]{1,2}\d{1,2}[A-Z]?)\b'
    postcode = None
    clean_parts = []

    for part in parts:
        part = part.strip()
        if re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}|[A-Z]{1,2}\d{1,2}[A-Z]?)$', part, re.IGNORECASE):
            postcode = part.upper().strip()
        elif part.lower() not in skip_values:
            clean_parts.append(part)

    street = clean_parts[0] if len(clean_parts) > 0 else None
    area = clean_parts[1] if len(clean_parts) > 1 else None

    # extract embedded postcode from area and street where its  lie "Hampstead NW3" etc.
    if area:
        embedded = re.search(postcode_pattern, area, re.IGNORECASE)
        if embedded:
            if not postcode:
                postcode = embedded.group(0).upper().strip()
            area = area[:embedded.start()].strip()
            area = area if area else None

    if street and not postcode:
        embedded = re.search(postcode_pattern, street, re.IGNORECASE)
        if embedded:
            postcode = embedded.group(0).upper().strip()
            street = street[:embedded.start()].strip()
            street = street if street else None

    return {"street": street, "area": area, "postcode": postcode}

def fill_missing_from_street_lookup(df):
    lookup = df.dropna(subset=["street", "area", "postcode"]).drop_duplicates("street").set_index("street")
    missing_area = df["area"].isna() & df["street"].notna() & df["street"].isin(lookup.index)
    df.loc[missing_area, "area"] = df.loc[missing_area, "street"].map(lookup["area"])
    missing_postcode = df["postcode"].isna() & df["street"].notna() & df["street"].isin(lookup.index)
    df.loc[missing_postcode, "postcode"] = df.loc[missing_postcode, "street"].map(lookup["postcode"])
    return df

def avg_crime_rate(lat, lon):
    url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lon}&date=2024-11"
    response = requests.get(url)
    if response.status_code == 200:
        return len(response.json())
    else:
        return np.nan

#split out for reuse:
def haverstine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


#https://www.doogal.co.uk/london_stations
def enrich_with_location_data(df):
    stations_df = pd.read_csv("data/london_stations.csv")
    stations_df = stations_df.dropna(subset=["Latitude", "Longitude"])
    missing_count = 0
    total = len(df["latitude"])
    location_cache = {} 
    #euclidean distance in lat/lon space is not accurate but gives a rough estimate of proximity to stations
    for i, row in df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        lat, lon = row["latitude"], row["longitude"]
        cache_key = (round(lat, 5), round(lon, 5)) 

        if cache_key in location_cache:
            df.loc[i, "distance_to_tube"] = location_cache[cache_key]["distance_to_tube"]
            df.loc[i, "crime_count"] = location_cache[cache_key]["crime_count"]
            print(f"[{i}/{total}] Cache hit: {cache_key}")
            continue

        dists = np.sqrt(
            (stations_df["Latitude"] - row["latitude"])**2 + 
            (stations_df["Longitude"] - row["longitude"])**2
        )
        nearest = stations_df.iloc[dists.idxmin()]
        df.loc[i, "distance_to_tube"] = haverstine_distance(row["latitude"], row["longitude"], nearest["Latitude"], nearest["Longitude"])
        df.loc[i, "crime_count"] = avg_crime_rate(row["latitude"], row["longitude"])
        location_cache[cache_key] = {
            "distance_to_tube": df.loc[i, "distance_to_tube"],
            "crime_count": df.loc[i, "crime_count"]
        }
        missing_count += 1
        print(f"[{missing_count}/{total}] Crime Count Calculated: {df.loc[i, 'crime_count']}")
    print("Done Enriching With Location Data.")    
    return df

def clean_location(df):
    location_df = df["title"].apply(extract_location).apply(pd.Series)
    df = pd.concat([df, location_df], axis=1)
    df = fill_missing_from_street_lookup(df)

    return df





