# simple test
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

my_list = [('a', 2, 3),
           ('b', 5, 6),
           ('c', 8, 9),
           ('a', 2, 3),
           ('b', 5, 6),
           ('c', 8, 9)]
col_name = ['col1', 'col2', 'col3']

df = spark.createDataFrame(my_list, schema=col_name)

from AutoFeatures import AutoFeatures

Fs = AutoFeatures()

to_drop = Fs.to_drop(df, missing_thold=0.68, corr_thold=0.9, method="pearson", rotation=True,
                     display=True, tracking=True, cat_num=2)
print(to_drop)