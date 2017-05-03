#!/usr/bin/python

"""
Sentiment analysis on amazon reviews - part 3

1. Clean the reviews by converting them to lower case,
   splitting into tokens at whitespaces and characters
   that are not letters, and removing stop words
2. Compute Word2Vec vectors for each review.
3. Take the vocabulary of Word2Vec vectors and cluster them using kmeans
4. List two word clusters to understand how similar words are clustered.
5. Use the clustering to map reviews into cluster frequency vectors
6. Build a linear regression model for the ratings using the cluster frequency vectors
7. Using 5 folder cross validation to find the best model
8. Report the RMSE error on the training and test datasets

how to run:
/usr/spark2.0.1/bin/spark-submit --master yarn --deploy-mdde client sentiment_analysis_kmeans.py /path/to/input

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

    # return an array of tokens, all lowercase and with punctuation stripped
    tokenizer = RegexTokenizer(inputCol="reviewText",
                                outputCol="words",
                                pattern="[^\w]")

    # remove stopwords, use english as the default stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                outputCol="filteredwords",
                                stopWords = StopWordsRemover.loadDefaultStopWords('english'))

    word2vec = Word2Vec(inputCol=remover.getOutputCol(),
                        outputCol="wordVec",
                        vectorSize=300,
                        seed=42,
                        minCount=2)

    # read train data
    train = spark.read.json(traininputs)
    train = train.select('overall','reviewText').withColumnRenamed('label','reviewText')
    train = tokenizer.transform(train)
    train = remover.transform(train)
    train.cache()
    model = word2vec.fit(train)
    vocabulary = model.getVectors()
    vocabulary.cache()
    uniqueWords = vocabulary.keys()
    uniqueWords.cache()

    # Take the vocabulary of Word2Vec vectors and cluster them using kmeans
    kmeans = KMeans(featuresCol="vector", predictionCol="cluster", initMode="random") \
            .setK(1000).setSeed(1234)
    kmeansModel = kmeans.fit(vocabulary)
    # kmeansModel.save(sc,output + '/kmean_model')
    vocabularyRDD = kmeansModel.transform(vocabulary)

    # regParama: regularization parameter > = 0
    lr = LinearRegression(featuresCol=word2vec.getOutputCol(),
                            maxIter=5,
                            regParam=0.0)

    vocabularyRDD = vocabularyRDD.select('word','cluster').map(lambda row: (row['word'],str(row['cluster'])))
    dictionary = sc.broadcast(dict(vocabularyRDD.collect()))

    randomWord = random.choice(unique_words)
    syn = model.findSynonyms(randomWord, 5)

    vectors = []
    for word in unique_words:
        if word in dictionary:
            vectors.append(dictionary[word])
    return vectors

    lrModel = lr.fit(vocabularyRDD)
    lrPredict = lrModel.transform(vocabularyRDD)
    evaluator = RegressionEvaluator()
    print evaluator.evaluate(lrPredict))

if __name__ == '__main__':
    main()