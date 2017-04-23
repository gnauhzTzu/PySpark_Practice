
# for problem description refer to:
# https://courses.cs.sfu.ca/2016fa-cmpt-732-g5/pages/Assignment5B
# set the batch length of 10 seconds

import sys
import datetime
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


output = sys.argv[1]


def add_tuples(a, b):
    return tuple(sum(p) for p in zip(a, b))

def fit(data):
    xy_avg = data[0]/data[4]
    x_avg = data[1]/data[4]
    y_avg = data[2]/ata[4]
    x2_avg = data[3]/ata[4]
    XavgYavy = x_avg*y_avg
    Xavg_2 =pow(x_avg,2)
    slope = (xy_avg - XavgYavy)/(x2_avg - xavg2)
    intercept = y_avg - slope*x_avg
    return slope, intercept

def regression(rdd):
    if rdd.isEmpty():
        return()

    data = rdd.map(lambda l: l.split(' ')).map(lambda (x, y): (float(x), float(y)))
    data = data.map(lambda (x, y): (x*y, x, y, x*x, 1)).reduce(add_tuples)
    (slope, intercept) = fit(data)

    # data will not be head to a database
    rdd = sc.parallelize([(slope, intercept)], numSlices=1)
    rdd.saveAsTextFile(output + '/' + datetime.datetime.now().isoformat().replace(':', '-'))

def main():
    sc = SparkContext()
    ssc = StreamingContext(sc, 10) #batch of 10 sec
    lines = ssc.socketTextStream("271.0.0.1", 10001)
    lines.foreachRDD(regression)

    ssc.start()
    # a modest timeout so you don't actually create an infinitely-running
    ssc.awaitTermination(timeout=300)
    #ssc.stop()

if __name__ == "__main__":
    main()
