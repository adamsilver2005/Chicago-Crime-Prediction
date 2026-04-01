


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