#!/usr/bin/python

"""

This script does the same thing as temp_range.py but using SQL syntax

Problem:
1. What weather station had the largest temperature difference on each day?
2. Where was the largest difference between TMAX and TMIN?

Spark: 2.0.1
Python: 2.7.5

"""
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

warehouse_location = 'spark-warehouse'

spark = SparkSession \
    .builder \
    .appName("temp range sql") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

def main():
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

    df = spark.read.csv(path = inputs, header = True, schema = schema)
    df = df.where(df['QFLAG'].isNull()).cache()

    df.createOrReplaceTempView('df')

    dfMax = spark.sql("""
        SELECT station, date, value as max
        FROM df WHERE observation == 'TMAX'
        """).createOrReplaceTempView('dfMax')

    dfMin = spark.sql("""
        SELECT station, date, value as min
        FROM df WHERE observation == 'TMIN'
        """).createOrReplaceTempView('dfMin')

    del df
    spark.catalog.dropTempView("df")

    joinedDF = spark.sql("""
    SELECT t1.station, t1.date, (t1.max - t2.min) as range
    FROM dfMax t1
    JOIN dfMin t2
    ON t1.station = t2.station AND t1.date = t2.date
    """).createOrReplaceTempView('joinedDF')

    maxRangeDF = spark.sql("""
    SELECT date, MAX(range) AS maxrange FROM joinedDF
    GROUP BY date
    """).createOrReplaceTempView('maxRangeDF')

    result = spark.sql("""
    SELECT t1.station as station, t1.date as date, t2.maxrange as range
    FROM joinedDF t1
    JOIN maxRangeDF t2
    ON t1.date = t2.date AND t1.range = t2.maxrange
    """)

    outdata = result.rdd.map(lambda x: "%s %s %s" % (x[0], x[1], x[2]))
    outdata.coalesce(1).saveAsTextFile(output)

    spark.catalog.dropTempView("joinedDF")
    spark.catalog.dropTempView("maxRangeDF")
    spark.catalog.dropTempView("dfMin")
    spark.catalog.dropTempView("dfMax")
    del result

if __name__ == "__main__":
    main()