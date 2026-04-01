# Chicago Crime Prediction: End-to-End ML Pipeline
Python | SQL | BigQuery | PySpark | PyTorch | TensorFlow | Google Colab

A full end-to-end machine learning project built on 8.5 million rows of real Chicago Police Department crime data. The project covers the complete data science workflow, from cloud data warehousing and large-scale preprocessing to training two separate neural networks for different prediction tasks.

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dataset Columns](#dataset-columns)
- [Tech Stack](#tech-stack)
- [Stage 1: BigQuery and SQL Exploration](#stage-1-bigquery-and-sql-exploration)
- [Stage 2: PySpark Preprocessing](#stage-2-pyspark-preprocessing)
- [Stage 3a: PyTorch: Crime Type Classifier](#stage-3a-pytorch--crime-type-classifier)
- [Stage 3b: TensorFlow: Crime Count Regressor](#stage-3b-tensorflow--crime-count-regressor)
- [Stage 4: Evaluation and Comparison](#stage-4-evaluation-and-comparison)
- [Final Conclusion](#final-conclusion)
- [How to Run the Project](#how-to-run-the-project)


## Project Goal

This project uses Chicago Police Department crime data to build two distinct neural network models, each solving a different prediction problem:

- **Model 1 (PyTorch):** Given the time, location, and context of an incident, predict what *type* of crime occurred — a 10-class classification problem
- **Model 2 (TensorFlow/Keras):** Given a district and time period, predict how many crimes will occur — a regression problem

Beyond the models themselves, the project is designed to cover the full modern data science stack: cloud data warehousing with BigQuery, distributed preprocessing with PySpark, and neural network training with both major deep learning frameworks.


## Dataset

The dataset used is the **Chicago Crime Dataset**, a public dataset hosted on Google BigQuery:

```
bigquery-public-data.chicago_crime.crime
```

It contains reported incidents of crime recorded by the Chicago Police Department's CLEAR system from 2001 to present. The dataset is updated daily and contains over 8.5 million rows across 22 features.

**Key facts:**
- 8,520,964 total rows
- 22 features including crime type, location, time, arrest status, and district
- Date range: 2001 to present
- This project uses a filtered sample of 2015–2023 for model training

**Note:** The raw data is not included in this repository. See [How to Run the Project](#how-to-run-the-project) for instructions on pulling the data from BigQuery.


## Project Structure

```
chicago-crime-ml/
│
├── data/                               # Local data samples (git ignored)
│   ├── chicago_crime_sample.csv        # 200k row local sample from BigQuery
│   ├── model1_classification.csv       # Processed data for PyTorch model
│   └── model2_regression.csv          # Aggregated data for TensorFlow model
│
├── src/
│   ├── bigquery/
│   │   ├── 01_explore.sql              # SQL exploration queries
│   │   └── 02_load_to_local.py         # Pull sample from BigQuery to CSV
│   ├── pyspark/
│   │   └── 01_preprocess.py            # Full preprocessing pipeline
│   ├── pytorch/
│   │   └── 01_train_classifier.py      # Model 1 — crime type classifier
│   └── tensorflow/
│       └── 01_train_regressor.py       # Model 2 — crime count regressor
│
├── outputs/                            # Saved models and plots (git ignored)
├── README.md
├── requirements.txt
└── .gitignore
```


## Dataset Columns

| Column | Description |
|--------|-------------|
| unique_key | Unique identifier for each crime report |
| date | Full timestamp of when the crime occurred |
| primary_type | High-level crime category (e.g. THEFT, BATTERY) — Model 1 target |
| description | Detailed sub-category of the crime |
| location_description | Type of location where the crime occurred |
| arrest | Whether an arrest was made (true/false) |
| domestic | Whether the incident was domestic-related (true/false) |
| district | Chicago Police district number — used in Model 2 aggregation |
| ward | City ward number |
| community_area | Community area number |
| latitude | Latitude coordinate (randomized for person offences) |
| longitude | Longitude coordinate (randomized for person offences) |

**Engineered features added during preprocessing:**

| Feature | Description |
|---------|-------------|
| year | Extracted from date timestamp |
| month | Extracted from date timestamp |
| day_of_week | Extracted from date timestamp (1=Sunday, 7=Saturday) |
| hour_of_day | Extracted from date timestamp |
| is_rush_hour | Flag: 1 if 7–9am or 4–7pm |
| is_weekend | Flag: 1 if Saturday or Sunday |
| season | 1=Winter, 2=Spring, 3=Summer, 4=Fall |


## Tech Stack

| Tool | Role |
|------|------|
| BigQuery + SQL | Cloud data warehousing, exploration, and sampling |
| PySpark | Large-scale data cleaning and feature engineering |
| PyTorch | Crime type classifier — manual training loop, multi-class output |
| TensorFlow/Keras | Crime count regressor — declarative model.fit() approach |
| Google Colab (T4 GPU) | Model training on larger dataset with GPU acceleration |


## Stage 1: BigQuery and SQL Exploration

All exploration queries are in `src/bigquery/01_explore.sql`.

The full dataset contains **8,520,964 rows** across 22 features. Key findings from the SQL exploration:

**Null check results:**

| Column | Null Count |
|--------|-----------|
| primary_type | 0 |
| date | 0 |
| arrest | 0 |
| district | 47 |
| latitude | 95,518 |

`primary_type`, `date`, and `arrest` are fully clean. The 47 missing districts and ~1% missing coordinates are handled by PySpark in Stage 2.

**Crime type distribution (top 10):**

| Crime Type | Count | % of Dataset |
|------------|-------|--------------|
| THEFT | 1,810,047 | 21.24% |
| BATTERY | 1,552,084 | 18.21% |
| CRIMINAL DAMAGE | 968,567 | 11.37% |
| NARCOTICS | 766,629 | 9.00% |
| ASSAULT | 572,739 | 6.72% |
| OTHER OFFENSE | 532,132 | 6.24% |
| BURGLARY | 450,089 | 5.28% |
| MOTOR VEHICLE THEFT | 438,424 | 5.15% |
| DECEPTIVE PRACTICE | 394,785 | 4.63% |
| ROBBERY | 316,578 | 3.72% |

The dataset is imbalanced — THEFT alone accounts for 21% of all crimes. The top 10 crime types cover ~92% of the dataset and are the classes used for Model 1.

A random 5% sample filtered to 2015–2023 is exported as a 200k row CSV for local PySpark development using `src/bigquery/02_load_to_local.py`. Model training in Colab pulls a larger sample (1–2 million rows) directly from BigQuery.


