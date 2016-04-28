/**
 * A technical exercise from an IT company 
 */

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{SQLContext, Row} 
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{StopWordsRemover, HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils



object MLexercise {
	def main(args: Array[String]): Unit = {

		  val conf = new SparkConf().setAppName("Technical exercise")
		  val sc = new SparkContext(conf)

		  //Read data
		  val data = sc.textFile("train-tweets.txt")

	    //create a (temporary) testing data generated from the original (training) data with ratio 7:3 
	    val Array(lines, linesTest) = data.randomSplit(Array(0.7, 0.3), seed = 123L)

  	
	    val sqlContext = new SQLContext(sc)
		  import sqlContext.implicits._

		  //create a tuple (label, text) for each line of the training data
		  val pairs = lines.flatMap(l =>  {
				val label = l.split("\\W+",2).head match{
				case "positive" => 2.0 case "negative" => 0.0 case "neutral" => 1
				}
				val text = l.split("\\W+",2).tail.map(_.split("\\s+"))
						text.map( t => (label, t))
			})

			// generata a stopwords remover
			val remover = new StopWordsRemover().setInputCol("raw").setOutputCol("text")

			//generate dataframe for training data
			val temp = remover.transform(pairs.toDF("label","raw").cache()).select("label","text")
			val training  = temp.map( r => (r.getDouble(0), r.apply(1).toString() )).toDF("label","text").cache()


			//create a tuple (label, text) for each line of the testing data
			val pairsTest = linesTest.flatMap(l =>  {
				val label = l.split("\\W+",2).head match{
				case "positive" => 2.0 case "negative" => 0.0 case "neutral" => 1
				}
				val text = l.split("\\W+",2).tail.map(_.split("\\s+")) 
						text.map( t => (label, t))
			}) 
			//generate dataframe for testing data
			val temp2 = remover.transform(pairs.toDF("label","raw").cache()).select("label","text")
			val test  = temp2.map( r => (r.getDouble(0), r.apply(1).toString() )).toDF("label","text").cache()



			//Start processing and learning data
			//split the raw traing text documents into words, and  add a "words" column to the data frame
			val tokenizer = new Tokenizer()
		    .setInputCol("text")
		    .setOutputCol("words")

			val wordsTraining = tokenizer.transform(training)

			//converts the words column into feature vectors and add to data frame
			val hashingTF = new HashingTF()
		    .setInputCol("words")
		    .setOutputCol("rawFeatures")
		    .setNumFeatures(20)
			val featurizedTrainingData = hashingTF.transform(wordsTraining)


			val idf = new IDF().setInputCol(hashingTF.getOutputCol)
			  .setOutputCol("features")
			val idfModel = idf.fit(featurizedTrainingData)
			val rescaledTrainingData = idfModel.transform(featurizedTrainingData).select("label", "features")

			///// processing with testing data
			//val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

			val wordsTesting = tokenizer.transform(test)

			//converts the words column into feature vectors and add to data frame
			//val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
			val featurizedTestData = hashingTF.transform(wordsTesting)


			//val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
			val idfTestModel = idf.fit(featurizedTestData)
			val rescaledTestData = idfTestModel.transform(featurizedTestData).select("label", "features")





			//Fitting procedure

			// Run training algorithm  LR-LBFGS to build the model
			val labeledTrain = rescaledTrainingData.map ( r =>  LabeledPoint(r.getDouble(0), r(1).asInstanceOf[Vector] )).cache()
			val model = new LogisticRegressionWithLBFGS()
		    .setNumClasses(3)
		    .run(labeledTrain)

			//create tuple (label, prediction) for testing set
			val predictionAndLabels = rescaledTestData.map ( r =>  LabeledPoint(r.getDouble(0), r(1).asInstanceOf[Vector] )).cache()
					.map{ case LabeledPoint(label, features) =>
					val prediction = model.predict(features)
					(label, prediction)
			}
			predictionAndLabels.take(5)

			// create a metrics object
			val metrics = new MulticlassMetrics(predictionAndLabels)


			// Overall Statistics
			val precision = metrics.precision
			val recall = metrics.recall 
			val f1Score = metrics.fMeasure
			println("Overall Statistics")
			println(s"Precision = $precision")
			println(s"Recall = $recall")
			println(s"F1 Score = $f1Score")
			
			println("Statistics in details:")

			// Precision of each label
			val labels = metrics.labels
			labels.foreach { l => println(s"Precision($l) = " + metrics.precision(l))		}

			// Recall of each label
			labels.foreach { l => println(s"Recall($l) = " + metrics.recall(l))		}


			// F-measure of each label
			labels.foreach { l =>	println(s"F1-Score($l) = " + metrics.fMeasure(l))		}
			
			sc.stop()


	}
}
