from numpy import genfromtxt
import pyspark_setup # Need to import pre-requisits set up by pyspark.
from pyspark import SparkContext, SparkConf
csv_data = genfromtxt('home_data.csv', delimiter=',', names=True)

sc = SparkContext('local[3]')
homeRDD = sc.parallelize(csv_data)
print(homeRDD.count())
print(homeRDD.take(2))