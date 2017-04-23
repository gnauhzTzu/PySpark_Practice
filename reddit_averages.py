#!/usr/bin/python

"""
1. Parse the json file of reddit comments
2. output the count of comments and the average scores of each subreddit
3. produce JSON as output 

The difference between reduce and reducebykey:
ref: https://blog.cloudera.com/blog/2014/09/how-to-translate-from-mapreduce-to-apache-spark/

The difference between map and flatmap:
ref: http://www.dattamsha.com/2014/09/map-vs-flatmap-spark/
ref: https://www.linkedin.com/pulse/difference-between-map-flatmap-transformations-spark-pyspark-pandey
"""

import sys
import json
from itertools import izip_longest
from pyspark import SparkConf, SparkContext

inputs = sys.argv[1]
output = sys.argv[2]
# input = "/user/tzu/a1-reddit-2"
# output = "/user/tzu/output-1"

# create a function to sum the pairs
# like add_pairs((1,1,0), (2,5)) = (3,6, 0)
def add_pairs(a, b):
    # should by zip_longest in python 3
    return [x + y for x, y in izip_longest(a, b, fillvalue=0)]


def parseJson(line):
    return json.loads(line).map(lambda x: (x['subreddit'],(1, x['score'])))

conf = SparkConf().setAppName('reddit average')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

# print text.take(1)
# sc.version
subReddit  = text.map(parseJson)
subRedditReduce = subReddit.reduceByKey(add_pairs).cache()
# if want to output one file, add .coalesce(1)

averageScore = subRedditReduce.map(lambda (key, (count, score)): (key, float(score)/count))
averageScoreJson = averageScore.map(json.dumps)

averageScoreJson.saveAsTextFile(output)
