from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, RandomForestRegressor

def getTrainSet(path='home_data.csv', split=[0.7,0.3], sparkContext=None, applicationName="ML_Basic_Home_Data"):

    if sparkContext is None:
        print("No spark context. Creating a new one.")
        sparkContext = SparkSession\
            .builder\
            .appName(applicationName)\
            .getOrCreate()
    
    df = sparkContext.read.csv(path, header=True)
    df = df\
        .withColumn("price", df["price"].cast(DoubleType()))\
        .withColumn("sqft_living", df["sqft_living"].cast(DoubleType()))\
        .withColumn("grade", df["grade"].cast(IntegerType()))\
        .withColumn("bedrooms", df["bedrooms"].cast(IntegerType()))
                    
    print(df.columns)

    df.select("price", "sqft_living", "zipcode", "yr_built").show(5)

    # Get training sets

    (trainData, testData) = df.randomSplit(seed=123, weights=split)
    print("The total data is {}, the training is {} and the test is {}"\
        .format(df.count(), trainData.count(), testData.count()))
    
    return (trainData, testData)

def displayEvaluationResults(trainingData, testData, steps, algorithms, algorithmLabels = None):
    labels = None

    if trainingData is None or testData is None or algorithms is None or steps is None:
        raise ValueError("Train data or steps or algorithms lists are null")
    else:
        if algorithmLabels is None:
            labels = ["Algorithm " + str(x + 1) for x in range(0,len(algorithms))]
        else:
            if len(algorithms < len(algorithms)):
                raise ValueError("Must have at least as many labels as algorithns")
            else:
                labels = algorithmLabels
        
        for idx, alg in enumerate(algorithms):
            print("+++++%s Results+++++" % (labels[idx]))
            pipelineSteps = [step for step in steps]
            pipelineSteps.append(alg)
            simplePipeline = Pipeline(stages=pipelineSteps)

            model = simplePipeline.fit(trainingData)
            predictions = model.transform(testData)

            # Select example rows to display.
            predictions.select("prediction", "price", "features").show(5)

            # Select (prediction, true label) and compute test error
            evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)
            print("RMSE is: {}".format(rmse))

def getCategoricalTransformers(columns):
    result = []
    idxColumnSuffix = "Index"
    vectorColumnSuffix = "Vector"

    for column in columns:
        result.append(StringIndexer(inputCol=column, outputCol=column + "Index"))
        result.append(OneHotEncoder(inputCol=column + idxColumnSuffix, outputCol=column + vectorColumnSuffix))
    
    return result

def getPipelineTransformers(nonCategoricalColumns, categoricalColumns=[]):
    result = []
    columns = []

    if categoricalColumns is None or len(categoricalColumns) > 0:
        result.extend(getCategoricalTransformers(categoricalColumns))
        columns.extend([transformer.getOutputCol() for transformer in result])
    
    columns.extend(nonCategoricalColumns)
    result.append(VectorAssembler(inputCols=columns, outputCol="features"))

    return result

