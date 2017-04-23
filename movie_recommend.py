#!/usr/bin/python

"""
The goal of this script is to provide a simple movie recommendation
based on the user rating input
the input should be a txt file looks like:
"rate" "movie name"
9 Mad Max (1979)
rate range is 1 - 10

The training dataset and schema can be found from:
https://github.com/sidooms/MovieTweetings/tree/master/latest

Spark version: 2.0.1

"""

import sys
import re
from os.path import isfile
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, Rating

warehouse_location = 'spark-warehouse'

spark = SparkSession \
    .builder \
    .appName("movies recommendations") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext


def loadUserRatings(filePath):

    if not isfile(filePath):
        print "Can't find the file."
        sys.exit(1)
    # read user provided ratings as data frame instead of RDD
    f = open(filePath, 'r')
    result = [parseUserRatings(line) for line in f]
    # filter out none element
    result = [x for x in result if x is not None]
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return result


def parseUserRatings(line):
    pattern = re.compile('^(\\d+)\\s([\\w\\s:\(\)]+)')
    m = re.match(pattern, line)
    if m:
        rating = m.group(1)
        movieName = m.group(2).encode('ascii', 'ignore')
        return rating, movieName
    return None


def parseMovie(movie):
    line = movie.strip().split('::')
     # movie_id, movie_name
    return int(line[0]), line[1]


def parseRating(rating):
    line = rating.strip().split('::')
    # user_id, movie_id, rating
    return line[0], line[1], line[2]


def matchMovieId(ratingsInput, movies):
    lineList = []
    for line in ratingsInput:
        amatch = movies.filter(lambda x: line[1][:-1] in x[1])
        if not amatch.isEmpty():
            matchedMovieId = amatch.first()[0]
            # user_id, movie_id, rating
            # use 0 as the new input user id
            lineList.append((0, matchedMovieId, line[0]))
    result = sc.parallelize(lineList)
    return result


def main():
    movies = sc.textFile("/user/tzu/movies.dat").map(parseMovie).cache()
    ratings = sc.textFile("/user/tzu/ratings.dat").map(parseRating)
    output = sys.argv[2]
    # read in rate #movie_name
    ratingsInput = loadUserRatings(sys.argv[1])
    # return user_id, movie_id, rating
    ratingInputWithId = matchMovieId(ratingsInput, movies)

    # union the new input rating and the existing rating as the training set
    ratingUnion = ratings.union(inputRatingWithId)
    ratingUnion = ratingUnion.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2]))).cache()

    # rank: number of features
    rank = 8
    # number of iteration in ALS
    numIterations = 5
    model = ALS.train(ratingUnion, rank, numIterations)
    # broadcast the input rating to improve the performance

    ratingInputList = ratingInputWithId.map(lambda (user, product, rating): product)
    ratingInputList = sc.broadcast(ratingInputList.collect())
    unRatedMovie = movies.filter(lambda x: x[0] not in ratingInputList).map(lambda x: (0, x[0]))

    # movie_id, score
    predictions = model.predictAll(unRatedMovie).map(lambda pred: (pred.product, pred.rating))

    # inner join to get teh movie name of movie id
    predictions = predictions.join(movies).sortBy(lambda (movieid, (rating, name)): rating, ascending=False)

    outdata = predictions.map(lambda (movieid, (rating, name)):  "Recommend Movies: %s, Estimate Rating:%0.2f" % (title, score) )
    print("Successfully find the movies recommend to you.")
    outdata.saveAsTextFile(output)


if __name__ == "__main__":
    main()