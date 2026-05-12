import pandas as pd

def price_gbp(value):
    if (pd.isna(value) or str(value).strip() == '' or str(value).strip().lower() == 'poa'):
         return None
  
    cleaned = (
            str(value)
            .replace("£", "")
            .replace(",", "")
            .strip()
        ) 
    return float(cleaned)