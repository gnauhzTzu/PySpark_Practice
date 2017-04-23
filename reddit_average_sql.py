#!/usr/bin/python

"""
This is a SQLContext version of the previously script

how to run:
both input and output are file path
open pyspark shell
from reddit_average_sql import *
main()

"""
import sys
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("reddit average sql").getOrCreate()
sc = spark.sparkContext

def main(argv=None):
    if argv is None:
        inputs = sys.argv[1]
        output = sys.argv[2]

    # If I don't specify a schema for JSON input,
    # the data must be read twice: once to determine the schema,
    # and again to actually load the data. That can be a big cost.

    # for those fields don't specify in my schema,
    # they will be ignored when data was read,
    # so I only write the fields I need.
    schema = StructType([
        StructField('score', IntegerType(), False),
        StructField('subreddit', StringType(), False),

    ])

    comments = spark.read.json(inputs, schema = schema)

    # Another version without schema
    # comments = spark.read.json(input)
    averages = comments.select('subreddit', 'score').groupby('subreddit').avg().coalesce(1).cache()

    # Output result to JSON text
    # The number of output is small enough
    averages.write.save(output, format='json', mode='overwrite')


if __name__ == "__main__":
    main()