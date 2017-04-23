#!/usr/bin/python

"""
The goal is to improve the previous matrix multiply
when the matrix is sparse
please ref to https://rawgit.com/jnwang/cmpt733/master/Assignments/A1/A1-instruction.html#Task-B:-Scalable-Matrix-Multiplication-(Sparse-Matrix) for details

Use csr_matrix in scipy package to store the sparse matrix
as one of the advantages of the csr format is the efficient
arithmetic operations like CSR + CSR, CSR * CSR

the input sparse matrix format is like:
29:0.288848 83:0.803680 28:0.892304
54:0.084138 22:0.125568 58:0.099520

how to run:
two arguments (input and output file) should be provided together with script
/usr/spark2.0.1/bin/spark-submit matrix_multiply_sparse.py --master yarn --deploy-mode client /path/of/input /path/of/output

"""

import sys
import operator
from scipy.sparse import csr_matrix
from pyspark.sql import SparkSession

# get the input text to a matrix
def textToRdd(line):
    lspt = line.split(' ')
    pair = [i.split(':') for i in lspt]
    # pair gets the row location of the number
    # and the value of number
    # [[29,0.288848], [83,0.803680]. [28,0.892304]]
    return pair


def saveToCSR(pair):

    col = [int(i[0]) for i in pair]
    data = [float(i[1]) for i in pair]

    # as pair is a list of list, transfer to csr matrix need to
    # specific row location of all data is 0
    # a list has len(vector) 0
    row = [0, len(vector)]
    # provided size of the sparse matrix
    d = 100
    csr = csr_matrix((data,(row,col)), shape=(1, d))

    return csr


def outerProduct(csr):
    # Outer product of a csr matrix and the transpose of that matri
    csrt = csr.transpose()
    blocks = csrt.multiply(csr).todense()

    return blocks


def main():

    spark = SparkSession.builder.appName("matrix multiply sparse").getOrCreate()
    sc = spark.sparkContext

    inputs = sys.argv[1]
    output = sys.argv[2]

    # first map split the row and save each as float number
    # second map to get outer product of vector
    # last reduce to sum them together
    matrix = sc.textFile(inputs) \
                .map(textToRdd) \
                .map(saveToCSR) \
                .map(outerProduct) \
                .reduce(operator.add)

    # format the matrix as the input for output
    matrix_list =matrix.tolist()

    result = []
    for i in range(len(matrix_list)):
        row = list(matrix_list[i])
        line = ""
        for j in range(100):
            row_str += str(j) + ':' + str(row[i]) + ' '
        result.append(line)

    result = sc.parallelize(result)
    result.saveAsTextFile(output)

if __name__ == '__main__':
    main()