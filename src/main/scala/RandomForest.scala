import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

object RandomForest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("mltest")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    //load train df
    val df = spark.read.format("csv")
      .option("header", value = true)
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .option("mode", "DROPMALFORMED")
      .load("parkinsons.data")
      .drop("name") // drop name col
      .cache()

//    df.printSchema()
//    df.show(5)

    //split train and test data
    val seed = 1
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed)


    //columns for training
    // define the assembler to collect the columns into a new column with a single vector - "features"
    val assembler = new VectorAssembler()
      .setInputCols(df.columns.filter(_ != "status"))
      .setOutputCol("features")


    val randomForestClassifier = new RandomForestClassifier()
      .setLabelCol("status")
      .setFeaturesCol("features")
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)
      //.setNumTrees(10)


    val stages = Array(assembler, randomForestClassifier)
    val pipeline = new Pipeline().setStages(stages)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("status")
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("rawPrediction")

    //cross validation
    val paramMap = new ParamGridBuilder()
      .addGrid(randomForestClassifier.impurity, Array("gini", "entropy"))
      .addGrid(randomForestClassifier.numTrees, Array(5, 10, 15, 20))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 4, 5))
      .addGrid(randomForestClassifier.maxBins, Array(25))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramMap)
      .setNumFolds(5)

    val cvModel = cv.fit(train)
    val predictions = cvModel.transform(test)

//    val  model = pipeline.fit(train)
//    val predictions = model.transform(test)


    val predictionAndLabels = predictions.select("prediction", "status")
      .rdd
      .map(r => (r.getDouble(0),  r(1).asInstanceOf[Int].toDouble))


    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision(1.0)
    val recall = metrics.recall(1.0)

    val metricsROC = new BinaryClassificationMetrics(predictionAndLabels)
    val auc = metricsROC.areaUnderROC

    println("Precision: " + precision)
    println("Recall: " + recall)
    println("AUC: " + auc)

    // output to parquet file
    val parquet = spark.sparkContext.makeRDD(0 to 1).map(_ => (auc, precision, recall))
    val parquetSchema = List(
      StructField("Precision", DoubleType, nullable = true),
      StructField("Recall", DoubleType, nullable = true),
      StructField("AUC", DoubleType, nullable = true)
    )
    val parquetRow = Row(auc, precision, recall)
    val rowAsRdd = spark.sparkContext.parallelize(List(parquetRow))
    spark.createDataFrame(rowAsRdd, StructType(parquetSchema))
      .write.parquet("ml_test")


  }

}


