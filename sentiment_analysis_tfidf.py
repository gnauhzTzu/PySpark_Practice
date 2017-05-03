#!/usr/bin/python

"""
Sentiment analysis on amazon reviews - part 1

1. Clean the reviews by converting them to lower case,
   splitting into tokens at whitespaces and characters
   that are not letters, and removing stop words
2. Get bag of words and compute the TF-IDF vectors for each review
3. Use Normalizer to normalize TF-IDF vectors
4. Build a linear regression model for the ratings using the training dataset.
6. Use cross validation and grid search to find the best model
7. Report the RMSE error on the training and test datasets

how to run:
/usr/spark2.0.1/bin/spark-submit --master yarn --deploy-mdde client sentiment_analysis_tfidf.py /path/to/input

ref:
https://spark.apache.org/docs/latest/mllib-feature-extraction.html#normalizer

"""

import sys
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover, Normalizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("sentiment tfidf").getOrCreate()
sc = spark.sparkContext

traininputs = sys.argv[1]
testinputs = sys.argv[2]

def main():

    train = spark.read.json(traininputs)
    train = train.select('overall','reviewText').withColumnRenamed('label','reviewText')

    # include all the funcions below into a pipeline
    step = 0

    step += 1
    # return an array of tokens, all lowercase and with punctuation stripped
    tokenizer = RegexTokenizer(inputCol="reviewText",
                                outputCol=str(step) + "_tokenizer",
                                pattern="[^\w]")
    step += 1
    # remove stopwords, use english as the default stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                outputCol=str(step) + "_stopwords",
                                stopWords = StopWordsRemover.loadDefaultStopWords('english'))

    step += 1
    # numFeatures: The number  of the features
    hashingTF = HashingTF(inputCol=remover.getOutputCol(),
                            outputCol=str(step) + "_hashingTF",
                            numFeatures=2^20)

    step += 1
    idf = IDF(inputCol=hashingTF.getOutputCol(),
                outputCol=str(step) + "_idf")

    step += 1
    normalizer = Normalizer(p=2.0,
                            inputCol=idf.getOutputCol(),
                            outputCol=str(step) + "_normalizer")

    step += 1
    # regParama: regularization parameter > = 0
    lr = LinearRegression(featuresCol=normalizer.getOutputCol(), maxIter=25, regParam=0.0)


    # start to build the model pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, normalizer, lr])

    paramGrid = ParamGridBuilder() \
        .addGrid(tf.numFeatures, [1, 4, 8, 16, 32]) \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0]) \
        .build()

    evaluator = RegressionEvaluator(metricName="rmse")

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)

    # Run cross-validation, and choose the best set of parameters.
    model = cv.fit(train)

    # get RMSE on train data
    trainPredict = model.transform(train)
    trainRMSE = evaluator.evaluate(trainPredict)
    train.unpersist()

    # Evaluate the model on test data
    test = spark.read.json(testinputs)
    test = test.select('overall','reviewText').withColumnRenamed('label','reviewText')
    test = tokenizer.transform(test)
    test = remover.transform(test)
    test.cache()

    testPredict = model.transform(test)
    testRMSE = evaluator.evaluate(testPredict)
    test.unpersist()

    # print the model with best rmse
    result = "The best model with tf-idf having a Train RMSE: " \
          + str(trainRMSE) + "\n" + "Test RMSE: " + str(testRMSE) + "\n"
    print(result)

if __name__ == '__main__':
    main()