#!/usr/bin/python

"""
Goal of this script is to answer:
1. How to compute Jaccard Similarity?
2. How to evaluate an ER result?
3. How to further improve the quality of an ER result?

Process:
1. convert each record to a set then use Jaccard to quantify the similarity
2. avoid n^2 comparisons in implementing Jaccard
3. evaluate the quality of ER output.

ref: https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/spark-sql-udfs.html
"""

import re
import operator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("entity resolution").getOrCreate()
sc = spark.sparkContext

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        """
            Input: $df represents a DataFrame
                   $cols represents the list of columns (in $df) that will be concatenated and be tokenized

            Output: Return a new DataFrame that adds the "joinKey" column into the input $df

            Comments: The "joinKey" column is a list of tokens, which is generated as follows:
                     (1) concatenate the $cols in $df;
                     (2) apply the tokenizer to the concatenated string
            Here is how the tokenizer should work:
                     (1) Use "re.split(r'\W+', string)" to split a string into a set of tokens
                     (2) Convert each token to its lower-case
                     (3) Remove stop words
        """
        stopwords = self.stopWordsBC
        df = df.withColumn('joinKey', concat_ws(' ', cols[0], cols[1]))

        def joinKey(r, stopwords):
            tokens = re.split('\W+', r)
            tokenized = [x.lower() for x in tokens if t not in stopwords]
            return tokenized

        joinKeyUDF = udf(joinKey, ArrayType(StringType()))
        df = df.withColumn("joinKey", joinKeyUDF(concat_ws(' ', df[cols[0]], df[cols[1]]), stopwords))

        return df


    def filtering(self, df1, df2):
    """
    Input: $df1 and $df2 are two input DataFrames, where each of them
           has a 'joinKey' column added by the preprocessDF function

    Output: Return a new DataFrame $candDF with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
            where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
            Intuitively, $candDF is the joined result between $df1 and $df2 on the condition that
            their joinKeys share at least one token.

    Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons,
              you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function.
              Please come up with a more efficient algorithm.
    """
        def flatPair(id, joinKey):
            pairList = [(id, i) for i in joinKey]
            return pairList

        flatPairUDF = udf(flatPair, ArrayType(ArrayType(StringType())))
        flatDf1 = df1.withColumn('pairs', flatPairUDF(df1.id, df1.joinKey)).select(df1.pairs).flatMap()
        flatDf2 = df1.withColumn('pairs', flatPairUDF(df2.id, df2.joinKey)).select(df2.pairs).flatMap()

        schema1 = StructType([StructField("tokens1", StringType(), True), StructField("id1", StringType(), True)])
        schema2 = StructType([StructField("tokens2", StringType(), True), StructField("id2", StringType(), True)])

        invertedDF1 = spark.createDataFrame(flatDf1, schema1)
        invertedDF2 = spark.createDataFrame(flatDf2, schema2)

        joinedDf = (invertedDF1.join(invertedDF2, invertedDF1.tokens1 == invertedDF2.tokens2)
                               .select(invertedDF1.id1, invertedDF2.id2).dropDuplicates())

        joinedDf = (joinedDf.join(df1, df1.id == joinedDf.id1)
                            .select(joinedDf.id1, df1.joinKey,
                            joinedDf.id2).withColumnRenamed('joinKey','joinKey1'))

        joinedDf = (joinedDf.join(df2, df2.id == joinedDf.id2)
                            .select(joinedDf.id1, joinedDf.joinKey1,
                            joinedDf.id2, df2.joinKey).withColumnRenamed('joinKey','joinKey2'))
        return joinedDf


    def verification(self, candDF, threshold):
        """
            Input: $candDF is the output DataFrame from the 'filtering' function.
                   $threshold is a float value between (0, 1]

            Output: Return a new DataFrame $resultDF that represents the ER result.
                    It has five columns: id1, joinKey1, id2, joinKey2, jaccard

            Comments: There are two differences between $candDF and $resultDF
                      (1) $resultDF adds a new column, called jaccard, which stores the jaccard similarity
                          between $joinKey1 and $joinKey2
                      (2) $resultDF removes the rows whose jaccard similarity is smaller than $threshold
        """
         # Jaccard similarity function
        def jaccard(joinKey1, joinKey2, threshold):
            len1 = len(joinKey1)
            len2 = len(joinKey2)
            interset = joinKey1.intersection(joinKey2)
            len3 = len(interset)
            if ((len1 == 0) or (len2 == 0) or (len3 = 0)):
                return 0
            elif(float(len3) / (len1 + len2 - len3) < threshold):
                return 0
            else:
                return float(len3) / (len1 + len2 - len3)

        # get Jaccard index
        jaccardUDF = udf(jaccard, FloatType())
        resultDF = candDF.withColumn("jaccard", jaccardUDF(candDF.joinKey1, candDF.joinKey2, threshold))
        return resultDF.where(resultDF.jaccard >= threshold)

    def evaluate(self, result, groundTruth):
        """
            Input: $result is a list of  matching pairs identified by the ER algorithm
                   $groundTrueth is a list of matching pairs labeld by humans

            Output: Compute precision, recall, and fmeasure of $result based on $groundTruth, and
                    return the evaluation result as a triple: (precision, recall, fmeasure)

        """
        interset = result.intersection(groundTruth)
        countInterset = len(interset)
        lenr = len(result)
        leng = len(groundTruth)
        precision = float(countInterset)/lenr
        recall = float(countInterset)/leng

        fmeasure = 2.0 * precision * recall/(precision + recall)

        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print "Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count())

        candDF = self.filtering(newDF1, newDF2)
        print "After Filtering: %d pairs left" %(candDF.count())

        resultDF = self.verification(candDF, threshold)
        print "After Verification: %d similar pairs" %(resultDF.count())

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("data/sample/Amazon_Google_perfectMapping_sample") \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)