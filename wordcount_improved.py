#!/usr/bin/python

"""
1. Use regular expression to break lines
   into words, and convert all keys to lowercase,
   remove empty line, and normalize the unicode
2. Sort the words alphabetically
3. combine the data into one single partition as the data output is not big
"""

import operator
import re
import string
import sys
import unicodedata

from pyspark import SparkConf, SparkContext

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('word count improved')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

wordsep = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
words_list = text.flatMap(lambda line: wordsep.split(unicodedata.normalize('NFD', line.lower())))
words = filter(None, words_list).map(lambda w: (w, 1))

# cache the result
wordcount = words.reduceByKey(operator.add).coalesce(1).cache()

# output by word
outdata_word = wordcount.sortBy(lambda (w,c): w).map(lambda (w,c): u"%s %i" % (w, c))
outdata_word.saveAsTextFile(output+'/by-word')

# output by frequency
outdata_fre = wordcount.sortBy(lambda (w,c): (-c,w)).map(lambda (w,c): u"%s %i" % (w, c))
outdata_fre.saveAsTextFile(output+'/by-freq')