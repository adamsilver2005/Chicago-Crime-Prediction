# Chicago Crime Prediction: End-to-End ML Pipeline
Python | SQL | BigQuery | PySpark | PyTorch | TensorFlow | Google Colab

## Table of Contents

- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dataset Columns](#dataset-columns)
- [Stage 1: BigQuery Exploration](#stage-1-bigquery-exploration)
- [Stage 2: PySpark Preprocessing](#stage-2-pyspark-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model 1: LightGBM Crime Type Classifier](#model-1-lightgbm-crime-type-classifier)
- [Model 2: TensorFlow Crime Count Regressor](#model-2-tensorflow-crime-count-regressor)
- [Final Conclusion](#final-conclusion)
- [How to Run the Project](#how-to-run-the-project)


## Project Goal

Chicago averages hundreds of thousands of reported crimes per year. This project builds two end-to-end machine learning models on 8.5 million crime records from the City of Chicago (2001–present), using the full modern ML pipeline: data storage, data cleaning and preprocessing, and model training with neural networks and gradient boosting.

The two prediction tasks are:

1. Given features about a crime incident, classify which of the 10 most common crime types it belongs to (multi-class classification)

2. Given a police district and time period, predict how many crimes will occur that month (regression)

The project also compares the neural network against XGBoost on the regression task to investigate when deep learning is and isn't the right tool for this particular data set.


## Dataset

Source: bigquery-public-data.chicago_crime.crime (Google BigQuery Public Datasets)

The dataset contains every reported crime in the City of Chicago from 2001 to present, maintained by the Chicago Police Department. For this project, data was filtered to 2015–2023 to focus on the modern era of policing and to reduce the noise and variability from the older records.

| Property | Value |
|----------|-------|
| Full dataset size | ~8.5 million rows |
| Training sample pulled | ~1.36 million rows (60% random sample, 2015–2023) |
| After preprocessing | 1,231,143 rows (1st prediction task) |
| Aggregated rows | ~2,100 district-month pairs (2nd prediction task) |
| GCP Project ID | healthy-result-491819-e4 |

Key observations from initial exploration:

- THEFT is the most common crime type at ~22% of all records, and the dataset is significantly imbalanced across the 10 classes
- District crime volumes vary dramatically: District 2 averages ~35,000 crimes/year while District 20 averages ~9,000
- District 13 is absent from the dataset, this is because it was disbanded by the Chicago Police Department
- District 31 in 2002 contains only 9 records, which is likely a data artifact from before the district was fully operational
- Null rates are low on key columns; district and primary_type are reliable across all years

