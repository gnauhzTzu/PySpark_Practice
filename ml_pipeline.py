#!/usr/bin/python

"""
The goal is to get familar with parameter tuning with spark ml
please ref to
https://rawgit.com/jnwang/cmpt733/master/Assignments/A1/A1-instruction.html#Task-C:-Parameter-Tuning
A preliminary trainning model is provided

how to run:
two arguments (input and output file) should be provided together with script
/usr/spark2.0.1/bin/spark-submit --master yarn --deploy-mode client ml_pipeline.py /path/of/traininput /path/of/testinput
the output will be print at stdout
Spark: 2.0.1
Python: 2.7.5

"""

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

traininput = sys.argv[1]
testinput = sys.argv[2]

def main():

    spark = SparkSession.builder.appName("ml pipeline").getOrCreate()
    sc = spark.sparkContext

    # Read training data as a DataFrame
    trainDF = spark.read.parquet(traininput)

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    # hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(),
    #                        outputCol="features",
    #                        numFeatures=1000)
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(),
                            outputCol="features")
    # lr = LogisticRegression(maxIter=20, regParam=0.1)
    lr = LogisticRegression(maxIter=20)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # cross validation
    paramGrid = ParamGridBuilder()\
        .addGrid(hashingTF.numFeatures, [1000, 5000, 10000])\
        .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])\
        .build()
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)
    cvModel = cv.fit(trainDF)


    # Fit the pipeline to training data.
    model = pipeline.fit(trainDF)

    # Evaluate the model on testing data
    testDF = spark.read.parquet(testinput)
    prediction = model.transform(testDF)
    output = evaluator.evaluate(prediction)
    print("without parameter tuning output is:", output)

    cvPrediction = cvModel.transform(testDF)
    cvOutput = evaluator.evaluate(cvPrediction)
    print("areaUnderROC with parameter tuning:", cvOutput)

if __name__ == '__main__':
    main()

