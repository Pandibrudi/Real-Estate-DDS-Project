import pandas as pd
import re
from datetime import datetime, timedelta

today_date = datetime(2024, 11, 6)
yesterday_date = today_date - timedelta(days=1)

date_pattern_exact = r"^\d{2}/\d{2}/\d{4}$"
date_pattern_general =  r"\d{2}/\d{2}/\d{4}"

def classify_date(value):
    if pd.isna(value) or str(value).strip() == "":
        return {
            "date_field": None,
            "category": "unknown",
            "date_type": None
        }
    formatted_value = str(value).strip().lower()
    match = re.search(rf"(reduced|added)\s+on\s+({date_pattern_general})", formatted_value)

    if match:
        event_type = match.group(1)
        date_value = match.group(2)
        try:
            datetime.strptime(date_value, "%d/%m/%Y")
        except ValueError:
            return {
                "date_field": formatted_value,
                "category": "not_categorized",
                "date_type": "added_on"
            }
        return {
            "date_field": date_value,
            "category": "date_structure",
            "date_type": f"{event_type}_on",
        }
    if formatted_value == 'reduced today':
         return {
            "date_field": today_date.strftime("%d/%m/%Y"),
            "category": "date_structure",
            "date_type": "reduced_on",
        }
    if formatted_value == 'reduced yesterday':
          return {
            "date_field": yesterday_date.strftime("%d/%m/%Y"),
            "category": "date_structure",
            "date_type": "reduced_on",
        }
    if formatted_value == 'added today':
        return{
            "date_field": today_date.strftime("%d/%m/%Y"),
            "category" : "date_structure",
            "date_type" : "added_on",
        }
    if formatted_value == 'added yesterday':
           return{
            "date_field": yesterday_date.strftime("%d/%m/%Y"),
            "category" : "date_structure",
            "date_type" : "added_on",
        }
    
    if re.match(date_pattern_exact, formatted_value):
        return {
            "date_field": formatted_value,
            "category": "date_structure",
            "date_type": "added_on",
        }

    return {
        "date_field": formatted_value,
        "category": "not_categorized",
        "date_type": None
    }


def process_added_on_column(df, column="addedOn"):
    classified_rows =df[column].apply(classify_date)
    return pd.DataFrame(classified_rows.tolist(), index=df.index)




