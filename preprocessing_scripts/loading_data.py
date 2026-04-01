# Load the data from BigQuery on google cloud to my local system 

from google.cloud import bigquery 
import pandas as pd 


# project id and sample size
PROJECT_ID ="healthy-result-491819-e4"
SAMPLE_SIZE = 200000


client = bigquery.Client(project = PROJECT_ID)


QUERY = f"""
SELECT
    primary_type, 
    description,
    location_description,
    arrest,
    domestic,
    district,
    ward,
    community_area,
    EXTRACT(YEAR FROM date)     AS year,
    EXTRACT(MONTH FROM date)    AS month,
    EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
    EXTRACT(HOUR FROM date)     AS hour_of_day,
    latitude, 
    longitude,
FROM `bigquery-public-data.chicago_crime.crime`
WHERE date IS NOT NULL
    AND primary_type IS NOT NULL
    AND district IS NOT NULL
    AND EXTRACT(YEAR FROM date) BETWEEN 2015 AND 2023
    AND RAND() < 0.05 
LIMIT {SAMPLE_SIZE}
"""


print(f"RUnning BigQuery query (pulling up to {SAMPLE_SIZE:,} rows)...")

data = client.query(QUERY).to_dataframe()



print(f"loaded {len(data):,} rows")
print(f"Columns: {list(data.columns)}")
print(f"\nCrime type distribution (top 10):")
print(data["primary_type"].value_counts().head(10)) 



# save locally
data.to_csv("C:/Users/Ilike/OneDrive/Year 3/Personal Projects/Chicago-Crime-Prediction/data/chicago_crime_sample.csv", index = False)

print("\nSaved to data/chicago_crime_sample.csv")

