#!/usr/bin/python

"""
1. Parse the json file of reddit comments
2. tFind he comment with the highest score relative to the subreddit's average

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

outdata = commentScores.join(subRedditAvg).map(
# (u'gadgets', ((u'NancyGracesTesticles', 19.0), 2.9972087705138555))
    lambda (r, ((author, score), avg)): (r, author, float(score)/ avg)).sortBy(
    lambda x: x[2], ascending = False)

outdata.saveAsTextFile(output)