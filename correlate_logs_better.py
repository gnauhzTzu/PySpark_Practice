#!/usr/bin/python

"""

problem: Evaluate the NASA web server log. In web server logs, 
         the number of bytes transferred to a host might be correlated 
         with the number of requests the host makes. Evaluate that.
1. Calculate the correlation coefficient of each host's number of requests 
   and total bytes transferred
2. Use a more stable calculation definition
"""
import math
import re
import sys

from pyspark import SparkConf, SparkContext

inputs = sys.argv[1]
output = sys.argv[2]


# adds two tuples of any length pairwise
def add_tuples(a, b):
    return tuple(sum(p) for p in zip(a, b))


conf = SparkConf().setAppName('correlate logs better')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

# use a regular expression disassemble the log lines
linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")

# check for each line in text, if there's a match
# in the result list, filter out all None, 0, False and ''
filteredLine = text.map(lambda line: linere.match(line)).filter(None)
# is equivalent to
# result = re.match(pattern, string)

# get host, bytes
hostBytesPair = filteredLine.map(lambda line: (line.group(1), (1, float(line.group(4)))))

# reduce by host

# If you use .collect() or .coalesce(1) to turn an RDD into no parallelism list
# please comment the size of the data to indicate that it's save to turn into a non-distributed collection
# first reduce by key to get the six number for each key
reducedPair = hostBytesPair.reduceByKey(lambda a, b: add_tuples(a, b)).cache()

hostSum = len(reducedPair.distinct().countByKey())
countSum = reducedPair.map(lambda x : x[1][0]).sum()
bytesSum = reducedPair.map(lambda x : x[1][1]).sum()

countMean = float(countSum/hostSum)
bytesMean = float(bytesSum/hostSum)

reducedPair2 = reducedPair.map(lambda (host,(count, byte)): (count-countMean, byte-bytesMean)).cache()

xySum = reducedPair2.map(lambda (x, y): x*y).sum()
xSqrSum = math.sqrt(reducedPair2.map(lambda (x, y): pow(x,2)).sum())
ySqrSum = math.sqrt(reducedPair2.map(lambda (x, y): pow(y,2)).sum())

r = xySum / (xSqrSum * ySqrSum)
powerR = pow(r, 2)

outputData = sc.parallelize([('r',r),('r^2',powerR)]).coalesce(1)
outputData.saveAsTextFile(output)