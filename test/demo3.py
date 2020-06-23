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

indexCol = 'SK_ID_CURR'
labelCol = 'AMT_INCOME_TOTAL'

task = 'regression'

Fs = AutoFeatures()

# correlation selector
to_drop = Fs.corr_selector(data, index_col=indexCol, label_col=labelCol,
                           corr_thold=0.9, method="pearson", rotation=True,
                           display=False, tracking=False, cat_num=2)
print('corr_selector:')
print(to_drop)

# essential selectors (included: missing selector, unique selector, correlation selector)
to_drop = Fs.essential_drop(data, index_col=indexCol, label_col=labelCol,
                            missing_thold=0.68, corr_thold=0.9, method="pearson", rotation=True,
                            display=True, tracking=True, cat_num=2)
print('essential_drop:')
print(to_drop)

# ensemble selectors

to_drop = Fs.ensemble_drop(data, index_col=indexCol, label_col=labelCol, task=task)
print('ensemble_drop:')
print(to_drop)
