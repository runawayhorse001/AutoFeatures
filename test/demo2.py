from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# from PySparkAudit import dtypes_class, hist_plot, bar_plot, freq_items,feature_len
# from PySparkAudit import dataset_summary, rates, trend_plot

# path = '/home/feng/Desktop'

from AutoFeatures import AutoFeatures

# load dataset
data = spark.read.csv(path='../data/credit_example.csv',
                      sep=',', encoding='UTF-8', comment=None, header=True, inferSchema=True)
data = data.fillna(0)

data.show(5)
Fs = AutoFeatures()

# correlation selector
to_drop = Fs.corr_selector(data, corr_thold=0.9, method="pearson", rotation=True,
                      display=False, tracking=False, cat_num=2)
print(to_drop)


to_drop = Fs.to_drop(data, missing_thold=0.68, corr_thold=0.9, method="pearson", rotation=True,
                     display=True, tracking=True, cat_num=2)
print(to_drop)