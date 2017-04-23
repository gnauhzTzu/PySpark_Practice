#!/usr/bin/python

"""
1. This is an improvement of prviews script when dealing with huge data
2. Parse the json file of reddit comments
3. Find he comment with the highest score relative to the subreddit's average

how to run:
spark-submit --master=yarn-cluster --executor-memory=4g \
--num-executors=4 --executor-cores=4 <this script name>.py \
/user/tzu/a1-reddit-2 /user/tzu/output-5

how to stop:
yarn application -kill <the application id>
"""

import sys
import json
from itertools import izip_longest
from pyspark import SparkConf, SparkContext

inputs = sys.argv[1]
output = sys.argv[2]


def add_pairs(a, b):
    # should by zip_longest in python 3
    return [x + y for x, y in izip_longest(a, b, fillvalue=0)]


conf = SparkConf().setAppName('relative score')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

parsedText = text.map(json.loads).cache()

commentScores = parsedText.map(lambda x: (x['subreddit'], (x['author'], float(x['score']))))

subRedditReduce  = parsedText.map(lambda x: (x['subreddit'],(1, x['score']))).reduceByKey(add_pairs)
subRedditAvg = subRedditReduce.map(lambda (r, (count, total)): (r, float(total)/count)).filter(
    lambda (r, avg): avg > 0)


# the improvement goes here:
# turn the average score into dict as it's relatively small
# then send it to cluster
subRedditAvg = dict(subRedditAvg.collect())

# Broadcast a read-only variable to the cluster for reading in distributed functions.
# The variable will be sent to each cluster only once.
subRedditAvg = sc.broadcast(subRedditAvg)

outdata = commentScores.map(
    lambda (r, (author, score)): (r, author, float(score)/ subRedditAvg.value[r])).sortBy(
    lambda x: x[2], ascending = False)

outdata.saveAsTextFile(output)