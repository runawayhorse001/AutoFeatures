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

print(data.toPandas().head(5))

indexCol = ['SK_ID_CURR']
labelCol = 'TARGET'

task = 'classification'

Fs = AutoFeatures()

# correlation selector
to_drop = Fs.corr_selector(data, index_col=indexCol, label_col=labelCol,
                           corr_thold=0.9, method="pearson", rotation=True,
                           display=False, tracking=False, cat_num=2)
print('corr_selector::{}'.format(to_drop))

# essential selector (included: missing selector, unique selector, correlation selector)
to_drop = Fs.essential_drop(data, index_col=indexCol, label_col=labelCol,
                            missing_thold=0.6, corr_thold=0.9, method="pearson", rotation=True,
                            display=True, tracking=True, cat_num=2)
print('essential_drop::{}'.format(to_drop))

# ensemble selector (ensemble selector is based on essential selector.)
to_drop = Fs.ensemble_drop(data, index_col=indexCol, label_col=labelCol, task=task, tracking=True)
print('ensemble_drop::{}'.format(to_drop))
