import pyspark_setup # Need to import pre-requisits set up by pyspark.
import home_data_helper as helper
from pyspark.sql import SparkSession

from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, RandomForestRegressor


# Get training sets
(trainData, testData) = helper.getTrainSet("home_data.csv")

#Train & Evaluate

stringifier = StringIndexer(inputCol="zipcode", outputCol="zipIndex")
oneHotter = OneHotEncoder(inputCol="zipIndex", outputCol="zipVector")
vectorizer = VectorAssembler(inputCols=["sqft_living", "zipVector"], outputCol="features")
glr = GeneralizedLinearRegression(labelCol="price", family="gaussian", link="identity", maxIter=10, regParam=0.3)
# when I add categorical I need to ensure bins are high enough.
#rf = RandomForestRegressor(labelCol="price", seed=1234)
rf = RandomForestRegressor(labelCol="price", seed=1234, numTrees=500, maxDepth=10, maxBins=100, minInstancesPerNode=5, featureSubsetStrategy="all")

helper.displayEvaluationResults(trainData, testData, helper.getPipelineTransformers(["sqft_living", "bedrooms"],["zipcode", "grade"]), [glr, rf])