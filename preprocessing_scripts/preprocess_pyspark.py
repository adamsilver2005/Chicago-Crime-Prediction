


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd



# start the spark session
spark = SparkSession.builder \
    .appName("ChicagoCrimePreprocessing") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark session started")



# load data
data = spark.read.csv(
    "C:/Users/Ilike/OneDrive/Year 3/Personal Projects/Chicago-Crime-Prediction/data/chicago_crime_sample.csv",
    header= True,
    inferSchema= True
)

print(f"Loaded {data.count():,} rows, {len(data.columns)} columns")
data.printSchema()



# clean 
# drop rows with any nulls in our key columns
key_cols = [
    "primary_type", "district", "year", "month",
    "day_of_week", "hour_of_day", "arrest", "domestic"
]
df_clean = data.dropna(subset=key_cols)
print(f"After dropping nulls: {df_clean.count():,} rows")
 
# cast boolean-like columns to integers (0/1)
df_clean = df_clean \
    .withColumn("arrest_int",   F.col("arrest").cast(IntegerType())) \
    .withColumn("domestic_int", F.col("domestic").cast(IntegerType()))
 
# feature engineering 
# is it rush hour? (7-9am or 4-7pm)
df_clean = df_clean.withColumn(
    "is_rush_hour",
    F.when(
        (F.col("hour_of_day").between(7, 9)) |
        (F.col("hour_of_day").between(16, 19)),
        1
    ).otherwise(0)
)
 
# is it a weekend?
df_clean = df_clean.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
    # day_of_week: 1=Sunday, 7=Saturday in Spark
)
 
# season (1=winter, 2=spring, 3=summer, 4=fall)
df_clean = df_clean.withColumn(
    "season",
    F.when(F.col("month").isin([12, 1, 2]), 1)
     .when(F.col("month").isin([3, 4, 5]),  2)
     .when(F.col("month").isin([6, 7, 8]),  3)
     .otherwise(4)
)
 
# encode target for model 1 (crime type to integer label) 
# keep only the top N crime types to keep classification manageable
top_n = 10
top_types = (
    df_clean.groupBy("primary_type")
            .count()
            .orderBy(F.desc("count"))
            .limit(top_n)
            .select("primary_type")
)
df_model1 = df_clean.join(top_types, on="primary_type", how="inner")
print(f"Model 1 dataset (top {top_n} crime types): {df_model1.count():,} rows")
 
indexer = StringIndexer(inputCol="primary_type", outputCol="label")
df_model1 = indexer.fit(df_model1).transform(df_model1)
 
# build feature vector 
feature_cols = [
    "year", "month", "day_of_week", "hour_of_day",
    "district", "arrest_int", "domestic_int",
    "is_rush_hour", "is_weekend", "season"
]
 
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                           withMean=True, withStd=True)
 
pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(df_model1)
df_features = pipeline_model.transform(df_model1)
 
# save processed data as CSV for PyTorch / TensorFlow
# convert back to Pandas for easy loading in PyTorch/TF
cols_to_save = feature_cols + ["label", "primary_type"]
df_pd = df_features.select(cols_to_save).toPandas()
df_pd.to_csv("../../data/model1_classification.csv", index=False)
print("Saved model1_classification.csv")
 
# aggregate for model 2 (crime count per district/month)
df_model2 = df_clean.groupBy("district", "year", "month", "season") \
    .agg(F.count("*").alias("crime_count")) \
    .orderBy("year", "month", "district")
 
df_model2_pd = df_model2.toPandas()
df_model2_pd.to_csv("../../data/model2_regression.csv", index=False)
print("Saved model2_regression.csv")
 
print("\nPreprocessing complete!")
spark.stop()