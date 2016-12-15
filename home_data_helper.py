def getTrainSet(path='home_data.csv', split=[0.7,0.3], sparkContext):
    df = sparkContext.read.csv(path, header=True)
    df = df.withColumn("price", df["price"].cast(DoubleType()))\
                    .withColumn("sqft_living", df["sqft_living"].cast(DoubleType()))
                    
    print(df.columns)

    # Get training sets

    (trainData, testData) = df.randomSplit(seed=123, weights=[0.7,0.3])
    print("The total data is {}, the training is {} and the test is {}"\
        .format(df.count(), trainData.count(), testData.count()))
    
    return (trainData, testData)

def displayEvaluationResults(trainingData, testData, algorithms, steps):
    for alg in algorithms:
        print("+++++%s Results+++++" % (alg[1]))
        simplePipeline = Pipeline(stages=steps.append(alg[0]))
        model = simplePipeline.fit(trainingData)

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
