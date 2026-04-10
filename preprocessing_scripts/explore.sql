
-- Query 1: First look at the raw data
-- Just pulling 10 rows to see what the data looks like
-- columns, data types, and sample values
SELECT *
FROM `bigquery-public-data.chicago_crime.crime`
LIMIT 10;


-- Query 2: Total row count
-- Confirms how big the dataset is
-- Expecting ~8 million rows
SELECT COUNT(*) AS total_rows
FROM `bigquery-public-data.chicago_crime.crime`;


-- Query 3: Column names and data types
-- Useful for understanding what features we have
-- before we start cleaning and engineering
-- Alternative: just click the Schema tab in BigQuery
SELECT column_name, data_type
FROM `bigquery-public-data.chicago_crime`.INFORMATION_SCHEMA.COLUMNS
WHERE table_name = 'crime';


-- Query 4: Crime type distribution
-- Shows the count and percentage of each crime type
-- This is our TARGET VARIABLE for Model 1 (PyTorch)
-- Key insight: dataset is imbalanced — THEFT is 21%
-- while rare types like GAMBLING are only 0.17%
-- We will keep only the top 10 types for training
SELECT
  primary_type,
  COUNT(*) AS count,

  -- Calculate what percentage of total crimes each type represents
  -- SUM(COUNT(*)) OVER () is a window function that gets the grand total
  -- so we can divide each type's count by the overall total
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct

FROM `bigquery-public-data.chicago_crime.crime`
GROUP BY primary_type
ORDER BY count DESC
LIMIT 20;




-- Query 5: Crime count by year and district
-- This is our TARGET VARIABLE
-- Each row = one district in one year
-- Model will learn: given district + year + month,
-- predict how many crimes will occur
-- 
-- Key insights from results:
-- District counts vary hugely (district 2: 35k vs district 20: 9k)
-- District 13 is missing — disbanded by Chicago PD
-- District 31 in 2002 has only 9 crimes, likely a data artifact
-- PySpark will handle these outliers in Stage 2

SELECT
  EXTRACT(YEAR FROM date) AS year,
  district,
  COUNT(*) AS crime_count
FROM `bigquery-public-data.chicago_crime.crime`
WHERE district IS NOT NULL  -- exclude rows with no district assigned
GROUP BY year, district
ORDER BY year, district;



-- Query 6: Null check on key columns
-- Before cleaning we need to know how much data is missing
-- Any column with high nulls needs special treatment in PySpark
-- Columns checked are the ones we use as features or targets
-- in both models, nulls here = rows we cannot use for training
SELECT
  COUNTIF(primary_type IS NULL)   AS null_primary_type,  -- Model 1 target
  COUNTIF(date IS NULL)           AS null_date,           -- used to extract time features
  COUNTIF(district IS NULL)       AS null_district,       -- Model 2 target grouping
  COUNTIF(latitude IS NULL)       AS null_latitude,       -- location feature
  COUNTIF(arrest IS NULL)         AS null_arrest          -- contextual feature
FROM `bigquery-public-data.chicago_crime.crime`;




-- Query 7: Pull a random sample for local development
-- We cannot load all 8.5M rows into PySpark on a laptop
-- So we take a random 5% slice filtered to 2015-2023
-- 200k rows is enough to train both neural networks
-- and small enough to process quickly locally
--
-- We also extract time components here (year, month etc.)
-- because the raw date column is a full timestamp,
-- neural networks need these as separate numeric features
SELECT
  primary_type,
  description,
  location_description,
  arrest,
  domestic,
  district,
  ward,
  community_area,

  -- Break the timestamp into separate numeric features
  -- e.g. "2019-03-15 14:32:00" becomes year=2019, month=3, etc.
  EXTRACT(YEAR FROM date)         AS year,
  EXTRACT(MONTH FROM date)        AS month,
  EXTRACT(DAYOFWEEK FROM date)    AS day_of_week,  -- 1=Sunday, 7=Saturday
  EXTRACT(HOUR FROM date)         AS hour_of_day,

  latitude,
  longitude
FROM `bigquery-public-data.chicago_crime.crime`
WHERE date IS NOT NULL           -- drop rows with no timestamp
  AND primary_type IS NOT NULL   -- drop rows with no crime type (Model 1 target)
  AND district IS NOT NULL       -- drop rows with no district (Model 2 grouping)
  -- Recent years only — data quality improves over time
  AND EXTRACT(YEAR FROM date) BETWEEN 2015 AND 2023
  -- RAND() < 0.05 keeps roughly 5% of rows randomly
  AND RAND() < 0.05
ORDER BY RAND()   -- shuffle so the sample isn't ordered by date
LIMIT 200000;     -- safety cap — never pull more than 200k rows