# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:25:26 2022

@author: 65904
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('codalong_logreg').getOrCreate()

df = spark.read.csv(r'C:/Users/65904/Desktop/Online-Courses-master/Spark_and_Python_for_Big_Data_with_PySpark/9_titanic.csv', inferSchema=True, header=True)
df.show()
df.printSchema()

my_cols = df.select(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
df = my_cols.na.drop()


from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer


# OneHotEncoding
gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVec')

embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex', outputCol='EmbarkVec')


assembler = VectorAssembler(inputCols=['Pclass', 'SexVec', 'EmbarkVec', 'Age', 'SibSp', 'Parch', 'Fare'],
                           outputCol='features')

# Classifier
from pyspark.ml.classification import LogisticRegression
classifier = LogisticRegression(featuresCol='features', labelCol='Survived')


# Creating the pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[gender_indexer, embark_indexer, gender_encoder, 
                            embark_encoder, assembler, classifier])


train_data, test_data = df.randomSplit([0.7, 0.3])
fit_classifier = pipeline.fit(train_data)

# Evaluation
results = fit_classifier.transform(test_data) # automatically calls the predict column 'prediction'

from pyspark.ml.evaluation import BinaryClassificationEvaluator


my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')
results.select('Survived', 'prediction').show()
AreaUC = my_eval.evaluate(results)
AreaUC