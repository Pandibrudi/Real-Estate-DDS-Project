
from cleaning.location_extraction import enrich_with_location_data, enrich_postcodes
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
 #https://get-information-schools.service.gov.uk/  
 # types of schooles necessary here.  


def get_category(row):
    etab = str(row.get('EstablishmentTypeGroup (name)', ''))
    phase = str(row.get('PhaseOfEducation (name)', ''))
    special = str(row.get('SpecialClasses (name)', ''))
    
    if 'Independent' in etab:
        funding_type = 'Private'
    elif 'Special' in etab or special == 'Yes':
        funding_type = 'Special'
    else:
        funding_type = 'State'
        
    if 'Primary' in phase:
        education_phase = 'Primary'
    elif 'Secondary' in phase:
        education_phase = 'Secondary'
    elif '16 plus' in phase:
        education_phase = 'Sixth Form'
    else:
        education_phase = 'Other'
        
    return pd.Series([funding_type, education_phase], index=['FundingType', 'EducationPhase'])

#generally catchment radius change year on year, but we can estimate
def add_catchment_radius(df):
    df = df.copy()

    df["catchment_km"] = 1.0  
    df.loc[df["EducationPhase"] == "Primary", "catchment_km"] = 0.8
    df.loc[df["EducationPhase"] == "Secondary", "catchment_km"] = 2.0
    df.loc[df["EducationPhase"] == "Sixth Form", "catchment_km"] = 3.0

    return df

def clean_school_data():
    #london prefixes:
    keep_cols = [
    'URN', 
    'EstablishmentName', 
    'Postcode', 
    'EstablishmentTypeGroup (name)', 
    'PhaseOfEducation (name)', 
    'SpecialClasses (name)'
]
    est_school = pd.read_csv("data/schools_uk_2024.csv",usecols=keep_cols, encoding='latin1')
    london_pattern = r'^(E|EC|N|NW|SE|SW|W|WC)\d'
    mask = est_school['Postcode'].str.match(london_pattern, na=False)
    schools = est_school[mask].copy()
    schools = enrich_postcodes(schools, cache_path="data/school_postcode_cache.csv", column='Postcode')
    schools[['FundingType', 'EducationPhase']] = schools.apply(get_category, axis=1)
    schools = add_catchment_radius(schools)

    return schools



def add_ofsted_ratings(schools_df):
    schools_df = schools_df.copy()
    schools_df['URN']= schools_df['URN'].astype(str)

    ofsted = pd.read_csv(
        'data/ofsted_info_2024.csv',
        encoding='latin1',
        dtype={'URN': str},
        low_memory=False
    )

    ofsted.columns= ofsted.columns.str.strip()

    merged_df= schools_df.merge(
        ofsted[['URN', 'Overall effectiveness']],
        on='URN',
        how='left'
    )
    return merged_df



def add_catchment_features(df, schools_df):
    schools_clean = schools_df[
        ['latitude', 'longitude', 'Overall effectiveness',
         'FundingType', 'EducationPhase', 'catchment_km']
    ].copy()

    for col in ['latitude', 'longitude', 'catchment_km']:
        schools_clean[col] = pd.to_numeric(schools_clean[col], errors='coerce')
    schools_clean['Overall effectiveness'] = pd.to_numeric(
        schools_clean['Overall effectiveness'], errors='coerce'
    )
    schools_clean = schools_clean.dropna(subset=['latitude', 'longitude', 'catchment_km']).reset_index(drop=True)

    if len(schools_clean) == 0:
        print("CRITICAL: No schools after cleaning.")
        return df

    prop_mask= df['latitude'].notna() & df['longitude'].notna()
    props= df.loc[prop_mask].reset_index(drop=True)

    prop_rad= np.radians(props[['latitude', 'longitude']].values)
    school_rad= np.radians(schools_clean[['latitude', 'longitude']].values)

    tree= BallTree(school_rad, metric='haversine')
    max_r= schools_clean['catchment_km'].max() / 6371
    indices, distances = tree.query_radius(prop_rad, r=max_r, return_distance=True)

    catchment_km= schools_clean['catchment_km'].values
    ratings= schools_clean['Overall effectiveness'].values
    funding= schools_clean['FundingType'].values
    phase= schools_clean['EducationPhase'].values

    counts, mean_ratings, best_ratings = [], [], []
    state_primary_counts, state_primary_best = [], []
    has_outstanding = []

    for idx, dists in zip(indices, distances):
        dist_km= dists * 6371
        in_catchment= dist_km <= catchment_km[idx]
        valid_idx= idx[in_catchment]

        school_ratings = ratings[valid_idx]
        school_funding= funding[valid_idx]
        school_phase= phase[valid_idx]
        rated_mask= ~np.isnan(school_ratings)
        counts.append(len(valid_idx))
        rated = school_ratings[rated_mask]
        mean_ratings.append(rated.mean() if len(rated) > 0 else np.nan)
        best_ratings.append(rated.min()  if len(rated) > 0 else np.nan)  # 1=Outstanding
        has_outstanding.append(int((rated == 1).any()) if len(rated) > 0 else 0)

        sp_mask= (school_funding == 'State') & (school_phase == 'Primary')
        sp_idx= valid_idx[sp_mask]
        sp_rated= ratings[sp_idx]
        sp_rated= sp_rated[~np.isnan(sp_rated)]
        state_primary_counts.append(len(sp_idx))
        state_primary_best.append(sp_rated.min() if len(sp_rated) > 0 else np.nan)

    df.loc[prop_mask, 'schools_in_catchment']= counts
    df.loc[prop_mask, 'catchment_mean_ofsted'] = mean_ratings
    df.loc[prop_mask, 'catchment_best_ofsted']= best_ratings
    df.loc[prop_mask, 'catchment_has_outstanding']= has_outstanding
    df.loc[prop_mask, 'state_primary_in_catchment']= state_primary_counts
    df.loc[prop_mask, 'state_primary_best_ofsted']= state_primary_best

    return df


def enrich_data(df):
    df = enrich_postcodes(df)
    df = enrich_with_location_data(df)
    school_df = clean_school_data()
    school_df = add_ofsted_ratings(school_df)
    print(f"Total schools: {len(school_df)}")
    print(f"Schools with ratings: {school_df['Overall effectiveness'].notna().sum()}")
    df = add_catchment_features(df, school_df)

    school_df.to_csv("data/schools_with_ratings.csv", index=False)
    df.to_csv("data/properties_with_catchment_features.csv", index=False)


    return df