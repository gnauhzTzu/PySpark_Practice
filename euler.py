#!/usr/bin/python

"""
Use Spark + Python to estimate Euler's consant using a stochastic representation
1 . Generate random numbers in [0,1] until the total is more than 1:
    the number of random values we needed for that is an estimate of e.
    If we average over say 10000 iterations, we will get a value really close to e.

2. For each iteration, save the number when we get the total greater than 1,
   each number per line
3. repeat the iterations to get (iteration number, value)

how to run:
export YARN_CONF_DIR=/etc/hadoop/conf/
export HADOOP_CONF_DIR=/etc/hadoop/conf/
/usr/spark2.0.1/bin/pyspark
/usr/spark2.0.1/bin/spark-submit --master yarn --deploy-mode client euler.py 30000000 10

ref:
http://spark.apache.org/docs/latest/cluster-overview.html
http://spark.apache.org/docs/latest/running-on-yarn.html

Same problem used to be inplemented with Java + MapReduce, the process speed has
been compared below:

With Java + MapReduce
two sets of input files have been tested
first one is a single file with 100 lines, number is 30000000 per each line
second one is 100 files with 1 line in each file, number is also 30000000 in that line.

running with first one is faster than second one because of the I/O of 100 files
first one takes 93938ms to finish, and
GC time elapsed (ms)=859
CPU time spent (ms)=249770
second one takes 274906ms to finish, and
GC time elapsed (ms)=23848
CPU time spent (ms)=402200

With Python + Spark

"""
import random
import sys

from pyspark.sql import SparkSession

def countOfSum(iter):
    count = 0
    # ref: https://stackoverflow.com/questions/31900124/random-numbers-generation-in-pyspark
    rand = random.Random()
    for i in xrange(iter):
        sum = 0.0
        while sum < 1 :
            sum = sum + rand.random()
            count += 1
    return count


def main():
    inputs_number = long(sys.argv[1])

    spark = SparkSession.builder.appName("euler estimation").getOrCreate()
    sc = spark.sparkContext

    # properly choose the partition number to
    # maximum parallelism.
    partitionsNum = 1000
    # partitionsNum = 100
    # partitionsNum = 10
    iterations = inputs_number/partitionsNum
    partitionArray = [iterations] * partitionsNum
    iterRDD = sc.parallelize(partitionArray, numSlices=partitionsNum)


    total = iterRDD.map(countOfSum).sum()
    print("Estimate of e value is : %0.8f" %(float(total)/inputs_number))


if __name__ == "__main__":
    main()