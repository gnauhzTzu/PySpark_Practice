#!/usr/bin/python

"""

problem: Evaluate the NASA web server log. In web server logs, 
         the number of bytes transferred to a host might be correlated 
         with the number of requests the host makes. Evaluate that.
1. Calculate the correlation coefficient of each host's number of requests 
   and total bytes transferred
2. For each host, add up the number of request and bytes transferred, 
   to form a data point(xi,yi).
3. Add the contribution of each data point to the six sums.
4. Calculate the final value of r.
5  Loads the logs from the input files, parses our useful info, 
   and calculates the correlation coefficient of each requesting host's 
   request count and total bytes transferred
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


# prepare for the correlation calculation
def sixSum(r):
    n = r[5]
    Sx = r[0]
    Sy = r[1]
    Sx2 = r[2]
    Sy2 = r[3]
    Sxy = r[4]
    correlate_r = (n*Sxy - Sx*Sy)/(math.sqrt(n*Sx2 - pow(Sx, 2)) * math.sqrt(n*Sy2 - pow(Sy, 2)))
    return correlate_r, pow(correlate_r,2)


conf = SparkConf().setAppName('correlate logs')
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
reducedPair = hostBytesPair.reduceByKey(lambda a, b: add_tuples(a, b))
reducedPair2 = reducedPair.map(lambda (host, (count, byte)): ('k', (count, byte, pow(count, 2), pow(byte, 2), count*byte, 1)))
# second reduce to get the sum of each six number for all keys
# reducedLast will be only one line
reducedLast = reducedPair2.reduceByKey(add_tuples).coalesce(1)

# get r and pow(r,2)
outputData = reducedLast.map(lambda (k, r): sixSum(r))

outputData.saveAsTextFile(output)