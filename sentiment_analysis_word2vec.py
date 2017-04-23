#!/usr/bin/python

"""
Sentiment analysis on amazon reviews - part 2

1. Clean the reviews by converting them to lower case,
   splitting into tokens at whitespaces and characters
   that are not letters, and removing stop words
2. Compute Word2Vec vectors for each review.
   Word2vec, an algorithm which converts a collection of words into a dictionary
   of multidimensional numerical representations
4. Build a linear regression model for the ratings using the training dataset.
6. Use cross validation and grid search to find the best model
7. Report the RMSE error on the training and test datasets

how to run:
/usr/spark2.0.1/bin/spark-submit --master yarn --deploy-mdde client sentiment_analysis_word2vec.py /path/to/input

"""

import sys
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, RegexTokenizer, StopWordsRemover
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("sentiment tfidf").getOrCreate()
sc = spark.sparkContext

traininputs = sys.argv[1]
testinputs = sys.argv[2]

def main():

    train = spark.read.json(traininputs)
    train = train.select('overall','reviewText').withColumnRenamed('overall','label')

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
    word2vec = Word2Vec(inputCol=remover.getOutputCol(),
                        outputCol=str(step) + "_word2vec",
                        vectorSize=100,
                        minCount=2)

    step += 1
    # regParama: regularization parameter > = 0
    lr = LinearRegression(featuresCol=word2vec.getOutputCol(),
                            maxIter=5,
                            regParam=0.0)

    # start to build the model pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, word2vec, lr])

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0]) \
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
    test = sqlCt.read.json(testinputs)
    test= test.select('overall','reviewText').withColumnRenamed('overall','label')
    test = tokenizer.transform(test)
    test = remover.transform(test)
    test.cache()

    testPredict = model.transform(test)
    testRMSE = evaluator.evaluate(testPredict)
    test.unpersist()

    # print the model with best rmse
    result = "The best model with Word2Vec having a Train RMSE: " \
          + str(trainRMSE) + "\n" + "Test RMSE: " + str(testRMSE) + "\n"
    print(result)

if __name__ == '__main__':
    main()