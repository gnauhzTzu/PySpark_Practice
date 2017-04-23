#!/usr/bin/python

"""
The goal of this script is to implement scalable multiple
of distributed matrix stored not in a single machine with spark
please ref to https://rawgit.com/jnwang/cmpt733/master/Assignments/A1/A1-instruction.html for details

the input matrix stored as distributed text file having 10^9 rows and 10 columns

how to run:
two arguments (input and output file) should be provided together with script
/usr/spark2.0.1/bin/spark-submit matrix_multiply.py --master yarn --deploy-mode client /path/of/input /path/of/output

"""

import sys
import operator

from pyspark.sql import SparkSession

# get the input text to a matrix
def textToRdd(line):
    vector = line.split(' ')
    vector = [float(i) for i in vector]
    return vector

def outerProduct(v):
    # Outer product of two same vectors
    assert isinstance(v, Vector), 'Need two vectors'

    n = len(v)
    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(v[i]*v[j])
        blocks.append(row)

    return blocks


def main():

    spark = SparkSession.builder.appName("matrix multiply").getOrCreate()
    sc = spark.sparkContext

    inputs = sys.argv[1]
    output = sys.argv[2]

    # first map split the row and save each as float number
    # second map to get outer product of vector
    # last reduce to sum them together
    matrix = sc.textFile(inputs) \
                .map(textToRdd) \
                .map(outerProduct) \
                .reduce(operator.add)
    result = sc.parallelize(matrix)

    result.saveAsTextFile(output)

if __name__ == '__main__':
    main()