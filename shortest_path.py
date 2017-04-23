#!/usr/bin/python

"""
Use a modification of Dijkstra's algorithm that can
be parallelized to find the shortest path in a graph
The graph will be representedby listing the outgoing edges of each node
nodes will be labelled with integers

input format(node numbers):
1: 3 5
2: 4
3: 2 5
4: 3
5:
6: 1 5

Problem:
1. Find the shortest path from <a input integer> to <a input integer>

output format should be like:

node 1: no source node, distance 0
node 3: source 1, distance 1
node 5: source 1, distance 1
node 2: source 3, distance 2
node 4: source 2, distance 3

How to run:
spark-submit --master yarn --deploy-mdde client shortest_path.py /path/to/input /path/to/output 1 4
Spark: 2.0.1
Python: 2.7.5

"""

import sys
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

inputs = sys.argv[1]
output = sys.argv[2]
startNode = sys.argv[3]
endNode = sys.argv[4]

def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if not (is_intstring(startNode) & is_intstring(endNode)):
    sys.exit("Argument 3 and 4 must be integers.")

def graphToKeyValuePair(line):
    lineSplit = line.split(':')
    # if one node has no outgoing edges
    # or one edge has not node number
    # it'll be save to remove them
    if not all(lineSplit):
        return []
    edgeSplit = filter(None, lineSplit[1].split(' '))
    result = []
    for i in edgeSplit:
        result.append((lineSplit[0], i))
    return result


def main():
    # ref: https://sparkour.urizone.net/recipes/understanding-sparksession/
    warehouse_location = 'spark-warehouse'

    spark = SparkSession \
        .builder \
        .appName("shortest path") \
        .config("spark.sql.warehouse.dir", warehouse_location) \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext

    # ref: http://stackoverflow.com/questions/34294143/how-can-i-return-an-empty-null-item-back-from-a-map-method-in-pyspark
    edgesMap = sc.textFile(inputs).flatMap(graphToKeyValuePair)

    # first check if the input start node exist
    if edgesMap.filter(lambda x: x[0] == startNode).count() == 0:
        sys.exit("Couldn't find the start node in the graph.")

    if edgesMap.filter(lambda x: x[1] == endNode).count() == 0:
        sys.exit("Couldn't find the destination node in the graph.")

    schema1 = StructType([StructField('from', StringType(), True),
        StructField('to', StringType(), True)])

    # same as toDF()
    edgesDF = spark.createDataFrame(edgesMap, schema1).cache()
    edgesDF.createOrReplaceTempView('edgesDF')

    schema2 = StructType([
        StructField('node', StringType(), False),
        StructField('source', StringType(), False),
        StructField('distance', IntegerType(), False),
    ])

    knownPath = sc.parallelize([[startNode,'no source node',0]])
    knownPath = spark.createDataFrame(knownPath,schema2)
    knownPath.cache()

    for i in range(6):
        knownPath.createOrReplaceTempView('previousKnownPath')

        currentPath = spark.sql("""
            SELECT t2.to AS node,
            t2.from AS source, (t1.distance + 1) AS distance
            FROM
            previousKnownPath t1
            LEFT JOIN
            edgesDF t2
            ON (t2.from = t1.node)
            ORDER BY t2.to
        """)

        currentPath.createOrReplaceTempView('currentPath')

        # find duplicate nodes between currentpath and previously path
        # remove it from currentPath
        duplicatePath = spark.sql("""
            SELECT t1.*
            FROM
            currentPath t1
            INNER JOIN
            previousKnownPath t2
            ON (t1.node = t2.node)
        """)

        currentPath = currentPath.subtract(duplicatePath)

        result = knownPath.unionAll(currentPath).rdd.map(
            lambda x: "node %s: source %s, distance %i" % (x[0], x[1], x[2])).coalesce(1)
        result.saveAsTextFile(output + '/iter-' + str(i))

        # check if we found the destination node
        if currentPath[currentPath['node'] == endNode].count() > 0:
            break

        # if not, update knownPath and keep search
        knownPath = currentPath

    # provide a path based on the loop result
    if currentPath[currentPath['node'] == endNode].count() = 0:
        sys.exit("Sorry! we couldn't find a path within 6 iteration.")

    outdata = []
    iterateNode = endNode
    for i in range(6):
        outdata.append(iterateNode)
        iterateNode = currentPath.filter(currentPath['endNode'] == iterateNode).select('source').first()[0]

    outdata = sc.parallelize(outdata)
    outdata.saveAsTextFile(output + '/path')

if __name__ == "__main__":
    main()