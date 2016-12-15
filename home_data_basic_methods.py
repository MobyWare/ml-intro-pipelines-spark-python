import pyspark_setup # Need to import pre-requisits set up by pyspark.
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("ML_Basic_Home_Data")\
    .getOrCreate()

from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, RandomForestRegressor

# Data Munging

df = spark.read.csv('home_data.csv', header=True)
df = df.withColumn("price", df["price"].cast(DoubleType()))\
                   .withColumn("sqft_living", df["sqft_living"].cast(DoubleType()))
                
#df.printSchema()

# Get training sets

(trainData, testData) = df.randomSplit(seed=123, weights=[0.7,0.3])
print("The total data is {}, the training is {} and the test is {}"\
      .format(df.count(), trainData.count(), testData.count()))

#Train & Evaluate

stringifier = StringIndexer(inputCol="zipcode", outputCol="zipIndex")
oneHotter = OneHotEncoder(inputCol="zipIndex", outputCol="zipVector")
vectorizer = VectorAssembler(inputCols=["sqft_living", "zipVector"], outputCol="features")
glr = GeneralizedLinearRegression(labelCol="price", family="gaussian", link="identity", maxIter=10, regParam=0.3)
rf = RandomForestRegressor(labelCol="price")
rfAdv = RandomForestRegressor(labelCol="price", numTrees=100, maxMemoryInMB=512, maxDepth=15)
for alg in [glr, rf, rfAdv]:
    print("+++++Method Start+++++")
    simplePipeline = Pipeline(stages=[stringifier, oneHotter, vectorizer, alg])
    model = simplePipeline.fit(trainData)

    #Print Reslts

    #testingData = vectorizer.transform(testData)
    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "price", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("RMSE is: {}".format(rmse))