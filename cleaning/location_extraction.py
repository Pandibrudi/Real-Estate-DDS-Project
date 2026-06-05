import os
import re
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

def enrich_postcodes(df, cache_path="data/postcode_cache.csv"):
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, index_col="postcode")
    else:
        cache = pd.DataFrame(columns=["postcode", "latitude", "longitude", "district", "ward", "constituency"]).set_index("postcode")
    mask = df["postcode"].notna()
    total = mask.sum()
    count = 0

    for idx, row in df.loc[mask].iterrows():
        count += 1
        postcode = row["postcode"]
        street = row.get("street")

        if postcode in cache.index:
            print(f"[{count}/{total}] Cache hit: {postcode}")
            for col in ["latitude", "longitude", "district", "ward", "constituency"]:
                df.loc[idx, col] = cache.loc[postcode, col] if col in cache.columns else None
            continue

        print(f"[{count}/{total}] Fetching: {street} {postcode}")
        data = fetch_postcode_data(postcode)

        for col in ["latitude", "longitude", "district", "ward", "constituency"]:
            df.loc[idx, col] = data.get(col)

        cache.loc[postcode] = data
        cache.to_csv(cache_path)
        time.sleep(0.1)

    missing_mask = df["postcode"].isna() & df["street"].notna()
    missing_total = missing_mask.sum()
    missing_count = 0

    for idx, row in df.loc[missing_mask].iterrows():
        missing_count += 1
        street = row["street"]
        print(f"[{missing_count}/{missing_total}] Nominatim fallback: {street}")
        
        data = fetch_by_street_nominatim(street)
        
        if data.get("postcode"):
            df.loc[idx, "postcode"] = data["postcode"]
        df.loc[idx, "latitude"] = data.get("latitude")
        df.loc[idx, "longitude"] = data.get("longitude")
        
        # fetch district/ward/constituency from postcodes.io using the new postcode - we tore in cache to stop spamming
        if data.get("postcode"):
            postcode_data = fetch_postcode_data(data["postcode"])
            for col in ["district", "ward", "constituency"]:
                df.loc[idx, col] = postcode_data.get(col)
            if df.loc[idx, "area"] is None and postcode_data.get("ward"):
                df.loc[idx, "area"] = postcode_data["ward"]
        
        time.sleep(1) 

    area_mask = df["area"].isna() & df["ward"].notna()
    df.loc[area_mask, "area"] = df.loc[area_mask, "ward"]

    print("Done The Enrichment Gathering.")
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
        R = 6371
        lat1, lon1 = np.radians(row["latitude"]), np.radians(row["longitude"])
        lat2, lon2 = np.radians(nearest["Latitude"]), np.radians(nearest["Longitude"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        df.loc[i, "distance_to_tube"] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
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
    df = enrich_postcodes(df)
    df = enrich_with_location_data(df)
    return df