#!/usr/bin/env python


"""
/* ******************************************************************************************
 * python 2.7.5
 * spark 2.0.1
 * hadoop 2.7.1
 * run with "arg00 arg0 arg1 arg2"
 * arg00: the location of spark-submit
 * arg0: the python script with location you want to submit to run
 * arg1: the path in hdfs of ratings.dat
 * arg2: the path in hdfs of movies.dat
 * arg3: the file path to your personal rating of movies file
 *       format:
 * /usr/spark2.0.1/bin/spark-submit /home/tzu/movie_recommend.py /user/tzu/ /home/tzu/learn-spark/training/machine-learning/personalRatings.txt
 *
 * ********************************************************************************************/
"""

import sys, string
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import split

# set up environment
conf = SparkConf().setAppName("movie_recommend")\
    .set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

def parseRatings(row):

    # format userId::movieId::rating::timestamp
    records = row.strip().split("::")
    return (int(records[0]), int(records[1]), float(records[2]))

def parseMovies(row):

    # format movieId::movieName
    records = row.strip().split("::")
    return int(records[0]), records[1].encode('ascii', 'ignore')

def parseUserRating(ratingsFile):
    """
    Pare the rating in the input rating file by users
    Format userId::movieId::rating::movieName
    """

    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[3] > 0,
        [parseRatings(row)[1] for row in f])
    f.close()

    if not ratings:
        print "Can not find eligible rate."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


def main():
    ratingFilePath = sys.argv[1]
    movieFilePath = sys.argv[2]
    userRatingFilePath = sys.argv[3]

    # load the raw rating and movie data
    rawRatings = sc.textFile(ratingFilePath)
    rawMovies = sc.textFile(movieFilepath)

    # parse the data
    parsedRatings = rawRatings.map(parseRatings).cache()
    parsedMovies = rawMovies.map(parseMovies).cache()
    parsedUserRating = parseUserRating(userRatingFilePath)
    parsedUserRating = sc.parallelize(parsedUserRating, 1).cache()



if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # load personal ratings
    myRatings = parseUserRating(sys.argv[2])
    myRatingsRDD = sc.parallelize(myRatings, 1)

    ratings = rawRatings.map(parseRatings)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(rawMovies.map(parseMovies).collect())
    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()
    print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

    # split ratings into train (60%), validation (20%), and test (20%) based on the
    # last digit of the timestamp, add myRatings to train, and cache them
    # training, validation, test are all RDDs of (userId, movieId, rating)

    numPartitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
      .values() \
      .repartition(numPartitions) \
      .cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()
    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()
    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # train models and evaluate them on the validation set
    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # compare the best model with a naive baseline that always returns the mean rating
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."

    # make personalized recommendations
    myRatedMovieIds = set([x[1] for x in myRatings])
    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]
    print "Movies recommended for you:"
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()