import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.common import flatten

import pyspark.sql.functions as F
from pyspark.mllib.stat import Statistics
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor


class AutoFeatures:
    """
    Auto feature selector for Machine Learning Modeling with PySpark. This class has four selectors:

    1. unique selector: identify the single unique features
    2. missing selector: identify missing values features with missing threshold
    3. collinear selector: identify collinear features with threshold
    4. low importance selector: identify low importance features with Gradient Boosting Machine(GBM)

    """

    def __init__(self):

        self.missing_drop = []
        self.unique_drop = []
        self.corr_drop = []
        self.total = None

    @classmethod
    def dtypes_class(cls, df_in):
        """
        Generate the data type categories: numerical, categorical, date and unsupported category.

        :param df_in: the input rdd data frame
        :return: data type categories

        >>> test = spark.createDataFrame([
                            ('Joe', 67, 'F', 7000, 'asymptomatic', 286.1, '2019-6-28'),
                            ('Henry', 67, 'M', 8000, 'asymptomatic', 229.2, '2019-6-29'),
                            ('Sam', 37,  'F', 6000, 'nonanginal', 250.3, '2019-6-30'),
                            ('Max', 56, 'M', 9000, 'nontypical', 236.4, '2019-5-28'),
                            ('Mat', 56, 'F', 9000, 'asymptomatic', 254.5, '2019-4-28')],
                            ['Name', 'Age', 'Sex', 'Salary', 'ChestPain', 'Chol', 'CreatDate']
                           )
        >>> test = test.withColumn('CreatDate', F.col('CreatDate').cast('timestamp'))
        >>> from PySparkAudit import dtypes_class
        >>> dtypes_class(test)
        (     feature       DataType
        0       Name     StringType
        1        Age       LongType
        2        Sex     StringType
        3    Salary       LongType
        4  ChestPain     StringType
        5       Chol     DoubleType
        6  CreatDate  TimestampType,
        ['Age', 'Salary', 'Chol'],
        ['Name', 'Sex', 'ChestPain'],
        ['CreatDate'], [])
        """

        # all data types in pyspark (for reference)
        # __all__ = [
        # "DataType", "NullType", "StringType", "BinaryType", "BooleanType", "DateType",
        # "TimestampType", "DecimalType", "DoubleType", "FloatType", "ByteType", "IntegerType",
        # "LongType", "ShortType", "ArrayType", "MapType", "StructField", "StructType"]

        # numerical data types in rdd DataFrame dtypes
        num_types = ['DecimalType', 'DoubleType', 'FloatType', 'ByteType', 'IntegerType', 'LongType', 'ShortType']
        # qualitative data types in rdd DataFrame dtypes
        cat_types = ['NullType', 'StringType', 'BinaryType', 'BooleanType']
        # date data types in rdd DataFrame dtypes
        date_types = ['DateType', 'TimestampType']
        # unsupported data types in rdd DataFrame dtypes
        unsupported_types = ['ArrayType', 'MapType', 'StructField', 'StructType']

        all_fields = [(f.name, str(f.dataType)) for f in df_in.schema.fields]

        all_df = pd.DataFrame(all_fields, columns=['feature', 'DataType'])

        # initialize the memory for the corresponding fields
        cls.num_fields = []
        cls.cat_fields = []
        cls.date_fields = []
        cls.unsupported_fields = []

        [cls.num_fields.append(item[0]) if item[1] in num_types else
         cls.cat_fields.append(item[0]) if item[1] in cat_types else
         cls.date_fields.append(item[0]) if item[1] in date_types else
         cls.unsupported_fields.append(item[0]) for item in all_fields]

        return all_df, cls.num_fields, cls.cat_fields, cls.date_fields, cls.unsupported_fields

    @classmethod
    def get_encoded_names(cls, df_in, categorical_cols):
        """
        get the encoded dummy variable names

        :param df_in: the input dataframe
        :param categorical_cols: the name list of the categorical columns
        :return: the name list of the encoded dummy variable for categorical columns
        """

        ind_names = [df_in.groupBy(c).count().sort(F.col("count").desc()).select(c).rdd.flatMap(lambda x: x).collect()
                     for c in categorical_cols]

        encodered_name = [categorical_cols[i] + '_' + j for i in range(len(categorical_cols)) for j in ind_names[i]]

        return encodered_name

    @classmethod
    def get_dummy(cls, df_in, index_col=None, categorical_cols=None, continuous_cols=None, label_col=None,
                  dropLast=False):
        """
        Get dummy variables and concat with continuous variables for ml modeling.

        :param df_in: the dataframe
        :param categorical_cols: the name list of the categorical data
        :param continuous_cols:  the name list of the numerical data
        :param label_col:  the name of label column
        :param dropLast:  the flag of drop last column
        :return: encoded dummy variable names and feature matrix

        :author: Wenqiang Feng
        :email:  von198@gmail.com

        >>> df = spark.createDataFrame([
                      (0, "a"),
                      (1, "b"),
                      (2, "c"),
                      (3, "a"),
                      (4, "a"),
                      (5, "c")
                  ], ["id", "category"])

        >>> index_col = 'id'
        >>> categorical_cols = ['category']
        >>> continuous_cols = []
        >>> label_col = []

        >>> mat = get_dummy(df,index_col,categorical_cols,continuous_cols,label_col)
        >>> mat.show()

        >>>
            +---+-------------+
            | id|     features|
            +---+-------------+
            |  0|[1.0,0.0,0.0]|
            |  1|[0.0,0.0,1.0]|
            |  2|[0.0,1.0,0.0]|
            |  3|[1.0,0.0,0.0]|
            |  4|[1.0,0.0,0.0]|
            |  5|[0.0,1.0,0.0]|
            +---+-------------+
        """

        # exclude index col and label col
        excluded = list(flatten([index_col, label_col]))

        if continuous_cols:
            continuous_cols = [col for col in continuous_cols if col not in excluded]
        else:
            continuous_cols = []

        if categorical_cols:
            categorical_cols = [col for col in categorical_cols if col not in excluded]
        else:
            categorical_cols = []

        indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categorical_cols]

        # default setting: dropLast=True
        encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                                  outputCol="{0}_encoded".format(indexer.getOutputCol()), dropLast=dropLast)
                    for indexer in indexers]

        assembler = VectorAssembler(inputCols=continuous_cols + [encoder.getOutputCol() for encoder in encoders],
                                    outputCol="features")

        pipeline = Pipeline(stages=indexers + encoders + [assembler])

        encodered_name = cls.get_encoded_names(df_in, categorical_cols)
        assemblered_name = continuous_cols + encodered_name

        model = pipeline.fit(df_in)
        data = model.transform(df_in)

        if index_col and label_col:
            # for supervised learning
            data = data.withColumn('label', F.col(label_col))
            out_data = data.select(*list(flatten([index_col, 'features', 'label'])))
        elif not index_col and label_col:
            # for supervised learning
            data = data.withColumn('label', F.col(label_col))
            out_data = data.select('features', 'label')
        elif index_col and not label_col:
            # for unsupervised learning
            out_data = data.select(*list(flatten([index_col, 'features'])))
        elif not index_col and not label_col:
            # for unsupervised learning
            out_data = data.select('features')

        return assemblered_name, out_data

    @classmethod
    def unique_selector(cls, data, tracking=False):
        """
        Unique selector: identify the single unique features

        :param data: input dataframe
        :param tracking: the flag for displaying CPU time, the default value is False
        :return unique_drop: The name list of the single unique features
        """

        start = time.time()

        nunique = data.agg(*[F.countDistinct(c) for c in data.columns]).rdd.flatMap(lambda x: x).collect()

        unique_drop = list(np.array(data.columns)[np.array(nunique) == 1])

        end = time.time()

        if tracking:
            print('Unique selector took = ' + str(end - start) + ' s')

        return unique_drop

    @classmethod
    def missing_selector(cls, data, missing_thold=0.6, display=False, tracking=False):
        """
        Missing selector: identify missing values features with missing threshold

        :param data: input dataframe
        :param missing_thold: threshold for missing values percentage
        :param display: the flag for displaying plots, the default value is False
        :param tracking: the flag for displaying CPU time, the default value is False
        :return: The name list of the missing values features above missing threshold
        """

        start = time.time()

        cls.total = data.count()

        rate_missing = [data.filter((F.col(c).isNull()) | (F.trim(F.col(c)) == '')).count() / cls.total
                        for c in data.columns]

        # self.rate_missing = rate_missing
        missing_drop = list(np.array(data.columns)[np.array(rate_missing) > missing_thold])

        end = time.time()

        if tracking:
            print('Missing selector took = ' + str(end - start) + ' s')

        if display:
            plt.figure(figsize=(10, 8))
            sns.distplot(rate_missing, bins=50, kde=False, rug=True, color='blue')
            plt.axvline(missing_thold, color='red', linestyle='--')
            plt.ylabel('Number of the features')
            plt.xlabel('Missing value percentage')
            plt.title('The missing value percentage Histogram')
            plt.show()

        return missing_drop

    @classmethod
    def corr_selector(cls, data, index_col=None, label_col=None, corr_thold=0.9, method="pearson", rotation=True,
                      display=False, tracking=False, cat_num=2):
        """
        collinear selector: identify collinear features with threshold

        :param data: input dataframe
        :param index_col: the name of the index column and the other columns you want to exclude
        :param label_col: the name of the label column
        :param corr_thold: threshold for collinear scores
        :param method: the method to use for computing correlation, supported: pearson (default), spearman
        :param rotation: the flag of rotate x-ticks
        :param display: the flag for displaying plots, the default value is False
        :param tracking: the flag for displaying CPU time, the default value is False
        :param cat_num: the number of the categorical feature (helping removing binary features)
        :return: The name list of the correlated values features above threshold
        """
        start = time.time()

        # numerical data types in rdd DataFrame dtypes
        num_types = ['DecimalType', 'DoubleType', 'FloatType',
                     'ByteType', 'IntegerType', 'LongType', 'ShortType']
        #
        num_fields = [f.name for f in data.schema.fields if str(f.dataType) in num_types]

        nunique = data.agg(*[F.countDistinct(c) for c in data.columns]).rdd.flatMap(lambda x: x).collect()

        flag = list(np.array(data.columns)[np.array(nunique) <= cat_num])

        # exclude binary, index and label cols
        if flag or index_col or label_col:
            if not isinstance(index_col, list):
                index_col = [index_col]
            excluded = list(dict.fromkeys(flag + index_col + [label_col]))
        else:
            excluded = []

        num_cols = [col for col in num_fields if col not in excluded]

        if len(num_cols) > 1:
            data = data.select(num_cols).na.drop()
        else:
            print("Only has one numerical feature!!! Don't need correlation selector.")
            exit(0)

        # fill nan with 0
        df_in = data.fillna(0)

        # convert the rdd data data frame to dense matrix
        col_names = df_in.columns
        features = df_in.rdd.map(lambda row: row[0:])

        # calculate the correlation matrix
        corr_mat = Statistics.corr(features, method=method)
        corr = pd.DataFrame(corr_mat)
        corr.index, corr.columns = col_names, col_names

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

        corr_drop = [column for column in upper.columns if any(upper[column].abs() > corr_thold)]

        end = time.time()

        if tracking:
            print('Correlation selector took = ' + str(end - start) + ' s')

        if display and corr_drop:
            dropped = corr[corr_drop]
            fig = plt.figure(figsize=(80, 40))  # Push new figure on stack
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

    @classmethod
    def importance_selector(cls, data, index_col, label_col, task, importance_thold=None, cumulative_thold=0.96,
                            missing_thold=0.6, corr_thold=0.9, method="pearson", rotation=True, n_train=5, top_n=20,
                            dropLast=False, display=False, tracking=False, cat_num=2):
        """
        importance selector: identify low feature importance features with threshold

        :param data: input dataframe
        :param index_col: the name of the index column and the other columns you want to exclude
        :param label_col: the name of the label column
        :param task: the ensemble model type, supported task "classification" or "regression"
        :param importance_thold: the threshold of the feature importance if missing will be auto calculated by the
               cumulative threshold
        :param cumulative_thold: the threshold of the cumulative feature importance, this will be used to determine
               the importance_thold when importance_thold is missing
        :param missing_thold: threshold for missing values percentage
        :param corr_thold: threshold for collinear scores
        :param method: the method to use for computing correlation, supported: pearson (default), spearman
        :param rotation: the flag of rotate x-ticks
        :param n_train: the numbers of train for average the feature importance
        :param top_n: the numbers for plot top_n highest feature importance
        :param dropLast: the flag of the drop last column during applying the OneHotEncoder
        :param display: the flag for displaying plots, the default value is False
        :param tracking: the number of the categorical feature (helping removing binary features)
        :param cat_num: the number of the categorical feature (helping removing binary features)
        :return: The name list of the dropped features with low feature importance
        """

        start = time.time()

        # essential drop list
        es_dropped = cls.essential_drop(data, index_col=index_col, label_col=label_col, missing_thold=missing_thold,
                                        corr_thold=corr_thold, method=method, rotation=rotation, display=display,
                                        tracking=tracking, cat_num=cat_num)

        data = data.drop(*es_dropped)
        data = data.fillna(0)
        data = data.na.fill('null')

        _, continuous_cols, categorical_cols, _, _ = cls.dtypes_class(data)
        dummy_name, features_dummy = cls.get_dummy(data, index_col, categorical_cols, continuous_cols, label_col,
                                                   dropLast)

        if task == 'classification':
            labelIndexer = StringIndexer(inputCol='label', outputCol='label_indexed')
            alg = GBTClassifier(labelCol='label_indexed',
                                featuresCol="features", maxIter=50)
            pipeline_model = Pipeline(stages=[labelIndexer, alg])
        elif task == 'regression':
            alg = GBTRegressor(labelCol='label',
                               featuresCol="features", maxIter=50)
            pipeline_model = Pipeline(stages=[alg])
        else:
            raise ValueError('Supported task "classification" or "regression"')

        # init feature importance
        feature_importance = np.zeros(len(dummy_name))

        for _ in range(n_train):

            (trainingData, _) = features_dummy.randomSplit([0.6, 0.4], seed=None)
            model = pipeline_model.fit(trainingData)

            if task == 'classification':
                importances = model.stages[1].featureImportances
            elif task == 'regression':
                importances = model.stages[0].featureImportances

            feature_importance += [a / n_train for a in importances]

        d = {'feature': dummy_name,
             'avg_importance': [a for a in feature_importance]}

        feature_importances = pd.DataFrame(d).sort_values('avg_importance', ascending=False)
        feature_importances['cumulative_importance'] = feature_importances['avg_importance'].cumsum()

        # set a flag to record whether we need plot cumulative importance
        plt_flag = False
        if not importance_thold:
            plt_flag = True # if we use the cumulative importance to determine the importance, we will plot
            importance_thold = feature_importances['avg_importance'][np.max(np.where(
                                         feature_importances['cumulative_importance'] < cumulative_thold))]

        # the numerical cols need to be dropped
        dropped_num = [dummy_name[i] for i in range(len(dummy_name))
                       if [a < importance_thold for a in feature_importance][i] and dummy_name[i] in continuous_cols]
        # the dummy names which importance higher than threshold
        keeped_dummy = ['_'.join(dummy_name[i].split('_')[:-1]) for i in range(len(dummy_name))
                        if [a > importance_thold for a in feature_importance][i]
                        and dummy_name[i] not in continuous_cols]
        # the dummy names which importance lower than threshold
        dropped_dummy = ['_'.join(dummy_name[i].split('_')[:-1]) for i in range(len(dummy_name))
                         if [a <= importance_thold for a in feature_importance][i]
                         and dummy_name[i] not in continuous_cols]

        # the categorical cols need to be dropped (not the dummy name). It's a little bit tricky to determine the
        # drop out the categorical cols.
        dropped_cat = [a for a in list(dict.fromkeys(dropped_dummy))
                       if (a not in list(dict.fromkeys(keeped_dummy)) and a in categorical_cols)]

        imp_drop = list(dict.fromkeys(es_dropped + dropped_num + dropped_cat))

        end = time.time()

        if tracking:
            print('All importance selector took = ' + str(end - start) + ' s')

        if display:
            if plt_flag:
                # plot the threshold of the cumulative importance
                plt.figure(figsize=(10, 8))
                ax = sns.lineplot(x=range(1, len(feature_importances) + 1), y="cumulative_importance",
                                  data=feature_importances, color='blue')

                cat_point = np.max(np.where(feature_importances['cumulative_importance'] < cumulative_thold))
                # plt.axhline(i_thre, ls='--')
                plt.text(cat_point + 5, cumulative_thold, (cat_point, cumulative_thold), color='red')
                plt.axvline(cat_point, color='red', linestyle='--')
                plt.xlabel('Number of the features')
                plt.show()

            # plot top_n feature importance
            plt.figure(figsize=(10, 8))
            plt_data = feature_importances[:top_n]
            ax = sns.barplot(x='avg_importance', y='feature', data=plt_data)
            ax.set_xlabel('avg_importance')
            plt.title("Top {} feature importance ".format(top_n), fontsize=20)
            plt.show()

        return imp_drop

    @classmethod
    def essential_drop(cls, data, index_col, label_col, missing_thold=0.6, corr_thold=0.9, method="pearson",
                       rotation=True, display=False, tracking=False, cat_num=2):
        """
        Essential drop (included: missing selector, unique selector, correlation selector) is all in one functions
        to identify the essential drop features.

        :param data: input dataframe
        :param index_col: the name of the index column and the other columns you want to exclude
        :param label_col: the name of the label column
        :param missing_thold: threshold for missing values percentage
        :param corr_thold: threshold for collinear scores
        :param method: the method to use for computing correlation, supported: pearson (default), spearman
        :param rotation: the flag of rotate x-ticks
        :param display: the flag for displaying plots, the default value is False
        :param tracking: the flag for displaying CPU time, the default value is False
        :param cat_num: the number of the categorical feature (helping removing binary features)
        :return: The name list of the to_drop features with essential  drop functions
        """

        start = time.time()

        unique_drop = cls.unique_selector(data, tracking=tracking)
        missing_drop = cls.missing_selector(data, missing_thold=missing_thold, display=display, tracking=tracking)
        corr_drop = cls.corr_selector(data, index_col=index_col, label_col=label_col, corr_thold=corr_thold,
                                      method=method, rotation=rotation, display=display, tracking=tracking,
                                      cat_num=cat_num)

        all_drop = unique_drop + corr_drop + missing_drop

        end = time.time()

        if tracking:
            print('The essential selector took = ' + str(end - start) + ' s')

        return list(dict.fromkeys(all_drop))

    @classmethod
    def ensemble_drop(cls, data, index_col, label_col, task, importance_thold=None, cumulative_thold=0.96,
                      missing_thold=0.6, corr_thold=0.9, method="pearson", rotation=True, n_train=5, top_n=20,
                      dropLast=False, display=False, tracking=False, cat_num=2):

        """
        Ensemble drop (based on essential drop, that is to say it has included the functionals of essential drop) is a
        method to identify the essential drop features based on ensemble ML model (GBM).

        :param data: input dataframe
        :param index_col: the name of the index column and the other columns you want to exclude
        :param label_col: the name of the label column
        :param task: the ensemble model type, supported task "classification" or "regression"
        :param importance_thold:  the threshold of the feature importance if missing will be auto calculated by the
               cumulative threshold
        :param cumulative_thold: the threshold of the cumulative feature importance, this will be used to determine
               the importance_thold when importance_thold is missing
        :param missing_thold: threshold for missing values percentage
        :param corr_thold: threshold for collinear scores
        :param method:  the method to use for computing correlation, supported: pearson (default), spearman
        :param rotation: the flag of rotate x-ticks
        :param n_train: the numbers of train for average the feature importance
        :param top_n: the numbers for plot top_n highest feature importance
        :param dropLast: the flag of the drop last column during applying the OneHotEncoder
        :param display: the flag for displaying plots, the default value is False
        :param tracking: the number of the categorical feature (helping removing binary features)
        :param cat_num: the number of the categorical feature (helping removing binary features)
        :return: The name list of the to_drop features with low feature importance
        """
        start = time.time()

        importance_drop = cls.importance_selector(data, index_col=index_col, label_col=label_col, task=task,
                                                  importance_thold=importance_thold, cumulative_thold=cumulative_thold,
                                                  missing_thold=missing_thold, corr_thold=corr_thold, method=method,
                                                  rotation=rotation, n_train=n_train, top_n=top_n, dropLast=dropLast,
                                                  display=display, tracking=tracking, cat_num=cat_num)

        end = time.time()

        if tracking:
            print('All ensemble selector took = ' + str(end - start) + ' s')

        return importance_drop


if __name__ == '__main__':

    from pyspark.sql import SparkSession

    spark = SparkSession \
        .builder \
        .appName("Python Spark regression example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    my_list = [('a', '1', 'f', 2, 3, 1, None, 0),
               ('b', '2', 'm', 5, 4, 1, None, 1),
               ('c', '3', 'm', 8, 9, 1, None, 0),
               ('d', '4', 'f', 2, 3, 1, None, 1),
               ('e', '5', 'm', 5, 5, 1, 4, 0),
               ('f', '6', 'm', 8, 9, 1, 4, 1)]
    col_name = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']

    df = spark.createDataFrame(my_list, schema=col_name)

    df.show()

    indexCol = ['col1', 'col3']
    labelCol = 'col7'  #
    categoricalCols = ['col2']
    task = 'classification'

    Fs = AutoFeatures()

    print('unique_selector')
    print(Fs.unique_selector(df))
    print('missing_selector')
    print(Fs.missing_selector(df))
    print('corr_selector')
    print(Fs.corr_selector(df, index_col=indexCol, label_col=labelCol))

    # get_dummy test
    names, dummied = Fs.get_dummy(df, index_col=indexCol, categorical_cols=categoricalCols, label_col=labelCol)

    print(names)
    dummied.show()

    # essential_drop test
    to_drop = Fs.essential_drop(df, index_col=indexCol, label_col=labelCol, method="pearson",
                                rotation=True, display=True, tracking=True, cat_num=2)
    print('essential_drop: {}'.format(to_drop))

    # importance threshold is determined by importance_thold or cumulative_thold, if importance_thold is missing then
    # the importance_thold will be calculated based on cumulative_thold
    print('importance_selector')
    print(Fs.importance_selector(df, index_col=indexCol, label_col=labelCol, importance_thold=0, task=task))

    to_drop = Fs.ensemble_drop(df, index_col=indexCol, label_col=labelCol, importance_thold=0, task=task)
    print('ensemble_drop::{}'.format(to_drop))

