#!/usr/bin/python

"""

The data can be found at
https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-ghcn

The data format is like:

US1FLSL0019,20130101,PRCP,0,,,N,
US1FLSL0019,20130101,SNOW,0,,,N

Problem:
1. What weather station had the largest temperature difference on each day?
2. Where was the largest difference between TMAX and TMIN?

Spark: 2.0.1
Python: 2.7.5

"""
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# In Spark > 2.0, SparkConf, SparkContext or SQLContext
# are encapsulated within SparkSession
warehouse_location = 'spark-warehouse'

# ref to http://spark.apache.org/docs/latest/sql-programming-guide.html#hive-tables
spark = SparkSession \
    .builder \
    .appName("temp range") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

def main(argv=None):
    if argv is None:
        inputs = sys.argv[1]
        output = sys.argv[2]

    schema = StructType([
        StructField('station', StringType(), False),
        StructField('date', StringType(), False),
        StructField('observation', StringType(), False),
        StructField('value', IntegerType(), False),
        StructField('MFLAG', StringType(), True),
        StructField('QFLAG', StringType(), True),
        StructField('SFLAG', StringType(), True),
        StructField('SFLAG2', StringType(), True),
    ])

    #read in as a dataframe
    df = spark.read.csv(path = inputs, header = True, schema = schema)
    # type(df)
    # df.printSchema()

    #get data with no quality issue
    df = df.where(df['QFLAG'].isNull()).cache()

    dfMax = df.filter(df['observation'] == 'TMAX').select(
        'station', 'date', 'value').withColumnRenamed(
        'value', 'max')

    dfMin = df.filter(df['observation'] == 'TMIN').select(
        'station', 'date', 'value').withColumnRenamed(
        'value', 'min')

    # remove uncessary dataframe
    del df

    df = dfMax.join(dfMin, ['station','date'], how='inner')
    df = df.select('station','date', (df['max']-df['min']).alias('range')).cache()

    # ref: https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
    windowSpec = Window.partitionBy(
        df['date']).orderBy(df['range'].desc())

    df = df.select('station','date','range',
        row_number().over(windowSpec).alias("rank"))

    result = df.filter(df['rank'] == 1).select('date', 'station', 'range')
    # result.show()

    # note: the dat is small enough to use coalesce(1)
    outdata = result.rdd.map(lambda x: "%s %s %s" % (x[0], x[1], x[2]))
    outdata.coalesce(1).saveAsTextFile(output)


if __name__ == "__main__":
    main()