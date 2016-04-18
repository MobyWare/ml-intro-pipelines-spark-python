import pyspark_setup
from pyspark import SparkContext, SparkConf

sc = SparkContext('local[3]')

print(sc.parallelize(range(1000)).count())