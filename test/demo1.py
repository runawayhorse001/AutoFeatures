# simple test
from AutoFeatures import AutoFeatures


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



df.show()

Fs = AutoFeatures()
indexCol = []
labelCol = []

to_drop = Fs.essential_drop(df, index_col=indexCol, label_col=labelCol, missing_thold=0.68, corr_thold=0.9,
                            method="pearson", rotation=True, display=True, tracking=True, cat_num=2)

print('essential dropped features:{}'.format(to_drop))

