# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:09:46 2022

@author: 65904
"""

# Initialising the SparkSession
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('logistic_regression').getOrCreate()

# Loading the dataset

df = spark.read.csv(r'C:/Users/65904/Desktop/Online-Courses-master/Spark_and_Python_for_Big_Data_with_PySpark/9_customer_churn.csv', inferSchema=True, header=True)
df.show()

# Examining the dataset
df.show()
df.describe().show()

# Transforming the dataframe into one accepted by PySpark
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Years', 'Num_Sites'],
                           outputCol='features')

output=assembler.transform(df)
output.show()


final_df = assembler.transform(df).select('features', 'Churn')
final_df.show()

# Train test split
train_data, test_data = final_df.randomSplit([0.7, 0.3])

# Creating the logistic regression model
from pyspark.ml.classification import LogisticRegression
classifier = LogisticRegression(featuresCol='features', labelCol='Churn', predictionCol='prediction')
fitted_classifier = classifier.fit(train_data)

# Evaluate
summary = fitted_classifier.summary
summary.predictions.describe().show()

# Evaluating using the testset
pred_vs_actual = fitted_classifier.evaluate(test_data)
pred_vs_actual.predictions.show()

# Area under the curve (ROC) evalutation
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Churn')
evaluator.evaluate(pred_vs_actual.predictions)


# Predicting using unlabeled data
# Loading the dataset
df_unlabeled = spark.read.csv(r'C:/Users/65904/Desktop/Online-Courses-master/Spark_and_Python_for_Big_Data_with_PySpark/9_new_customers.csv', inferSchema=True, header=True)
df_unlabeled.show()

# Transforming the dataframe into one accepted by PySpark
final_df_unlabeled = assembler.transform(df_unlabeled)
final_df_unlabeled.show()

# Creating a new model using the entire dataset
classifier_all = classifier.fit(final_df)
results = classifier_all.transform(final_df_unlabeled)
results.select('Company', 'prediction').show()


results = classifier_all.transform(final_df)
results.select('Company', 'prediction').show()