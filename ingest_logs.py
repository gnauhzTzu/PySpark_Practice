#!/usr/bin/python

"""
problem:
Read and parse the NASA web server log lines,
extract the requesting host, the datetime, the path, and the number of bytes
then store the data in a better format.
Use Parquet, a columnar storage format, with Spark SQL.
"""

import sys
import re
import datetime

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

conf = SparkConf().setAppName('nasa log ingest')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
text = sc.textFile(inputs)

def main(argv=None):
    if argv is None:
        inputs = sys.argv[1]
        outputdir = sys.argv[2]

    linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")

    request = text.map(lambda line: linere.split(line)).filter(None).map(
        lambda line: Row(hostname=line[1],
            timestamp=datetime.datetime.strptime(line[2], '%d/%b/%Y:%H:%M:%S'),
            path=line[3], size=float(line[4])))

    schema = StructType([
        StructField('hostname', StringType(), False),
        StructField('path', StringType(), False),
        StructField('size', FloatType(), False),
        StructField('timestamp', TimestampType(), False)
    ])

    request = sqlContext.createDataFrame(request, schema)
    request.write.format('parquet').save(outputdir)


if __name__ == "__main__":
    main()
