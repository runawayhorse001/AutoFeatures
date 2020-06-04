import os
import shutil
import time
import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pyspark.sql.functions as F
from pyspark.mllib.stat import Statistics


class AutoFeatures():
    """
    Auto feature selector for Machine Learning Modeling with PySpark.
    """

    def __init__(self):

        self.missing_drop = None
        self.unique_drop = None
        self.corr_drop = None
        self.missing_thold = None
        self.total = None

    def unique_selector(self, data, tracking=False):

        start = time.time()

        nunique = data.agg(*[F.countDistinct(c) for c in data.columns]).rdd.flatMap(lambda x: x).collect()
        # [data.na.drop(subset=[c]).select(c).distinct().count() for c in data.columns]

        self.unique_drop = list(np.array(data.columns)[np.array(nunique) == 1])

        end = time.time()

        if tracking:
            print('Unique selector took = ' + str(end - start) + ' s')

        return self.unique_drop

    def missing_selector(self, data, missing_thold=0.6, display=False, tracking=False):

        start = time.time()

        self.missing_thold = missing_thold
        self.total = data.count()

        rate_missing = [data.filter((F.col(c).isNull()) | (F.trim(F.col(c)) == '')).count() / self.total
                        for c in data.columns]

        # self.rate_missing = rate_missing
        self.missing_drop = list(np.array(data.columns)[np.array(rate_missing) > missing_thold])

        end = time.time()

        if tracking:
            print('Missing selector took = ' + str(end - start) + ' s')

        if display:
            plt.figure(figsize=(10, 8))
            sns.distplot(rate_missing, bins=50, kde=False, rug=True, color='blue');
            plt.show()

        return self.missing_drop

    def corr_selector(self, data, corr_thold=0.9, method="pearson", rotation=True,
                      display=False, tracking=False, cat_num=2):

        start = time.time()

        # numerical data types in rdd DataFrame dtypes
        num_types = ['DecimalType', 'DoubleType', 'FloatType',
                     'ByteType', 'IntegerType', 'LongType', 'ShortType']
        #
        num_fields = [f.name for f in data.schema.fields if str(f.dataType) in num_types]

        nunique = data.agg(*[F.countDistinct(c) for c in data.columns]).rdd.flatMap(lambda x: x).collect()
        # [data.na.drop(subset=[c]).select(c).distinct().count() for c in data.columns]

        flag = list(np.array(data.columns)[np.array(nunique) <= cat_num])

        num_cols = [col for col in num_fields if col not in flag]

        if len(num_cols) > 1:
            df_in = data.select(num_cols).na.drop()
        else:
            print("Only has one numerical feature!!! Don't need correlation selector.")
            exit(0)

        # convert the rdd data data frame to dense matrix
        col_names = df_in.columns
        features = df_in.rdd.map(lambda row: row[0:])

        # calculate the correlation matrix
        corr_mat = Statistics.corr(features, method=method)
        corr = pd.DataFrame(corr_mat)
        corr.index, corr.columns = col_names, col_names

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

        corr_drop = [column for column in upper.columns if any(upper[column].abs() > corr_thold)]

        self.corr_drop = corr_drop

        end = time.time()

        if tracking:
            print('Correlation selector took = ' + str(end - start) + ' s')

        if display:
            dropped = corr[corr_drop]
            fig = plt.figure(figsize=(40, 40))  # Push new figure on stack
            sns_plot = sns.heatmap(dropped, cmap="YlGnBu",
                                   xticklabels=dropped.columns.values,
                                   yticklabels=corr.columns.values,
                                   annot=True, fmt=".1g", linewidths=.25)
            plt.xlabel('Features to Drop', size=20)
            plt.ylabel('All Numerical Features', size=20)
            plt.title("Correlations Above Threshold {}".format(corr_thold), fontsize=20)
            if rotation:
                plt.xticks(rotation=90, fontsize=20)
                sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=20)
            plt.show()

        return corr_drop

    def to_drop(self, data, missing_thold=0.68, corr_thold=0.9, method="pearson", rotation=True,
                display=False, tracking=False, cat_num=2):

        start = time.time()

        self.unique_selector(data)
        self.missing_selector(data, missing_thold=missing_thold, display=display, tracking=tracking)
        self.corr_selector(data, corr_thold=corr_thold, method=method, rotation=rotation,
                           display=display, tracking=tracking, cat_num=2)

        all_drop = self.corr_drop + self.unique_drop + self.missing_drop

        end = time.time()

        if tracking:
            print('All selectors took = ' + str(end - start) + ' s')

        return all_drop# list(dict.fromkeys(all_drop))


if __name__ == '__main__':

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

    Fs = AutoFeatures()

    to_drop = Fs.to_drop(df, missing_thold=0.68, corr_thold=0.9, method="pearson", rotation=True,
                         display=True, tracking=True, cat_num=2)
    print(to_drop)

