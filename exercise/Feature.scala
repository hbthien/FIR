
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

val conf = new SparkConf().setAppName("An exercise with PR and F1")
val sc = new SparkContext(conf)

case class Sentence(category: String, text: String)
case class LabeledData(category: String, text: String, label: Double)

//val lines = sc.textFile("train-tweets.txt")  

val data = sc.textFile("train-tweets.txt")

//a (temporary) testing data generated from the original (training) data with ratio 7:3 
val Array(lines, linesTest) = data.randomSplit(Array(0.7, 0.3), seed = 123L)

val categoryMap = lines.map(_.split("\\W+",2)).map(x => Sentence(x(0),x(1))).
val cateList = categoryMap.map(x=>x.category).distinct.zipWithIndex.mapValues(x=>x.toDouble).collectAsMap

//val pos = lines.filter(_. startsWith ("positive"))
//val neg = lines.filter(_. startsWith ("negative"))
//val neu= lines.filter(_. startsWith ("neutral") 
val labeledData = categoryMap.map(x=>LabeledData(x.category, x.text, cateList.get(x.category).getOrElse(0.0))).
    toDF("category","text","label")

    
    

val pairs = lines.map(l => l.split("\\W+",2)).map(l=>Sentence(l(0).toDouble,l(1)))

//val pairs = lines.map(l => l.split("\\s+")(0)) //tach 1 cau thanh nhieu tu, va lay tu dau
//val pairs = lines.map(l => l.split("\\W+",2)) //tach 1 cau thanh 2 phan: 1/tu dau tien, 2/con lai

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

///////////////// IF USE STOPREMOVER ////////////
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
//generate dataframe for training data
val temp2 = remover.transform(pairs.toDF("label","raw").cache()).select("label","text")

val test  = temp2.map( r => (r.getDouble(0), r.apply(1).toString() )).toDF("label","text").cache()



//////////////////////////IF NOT USE STOPREMOVER///////////////

//create a tuple (label, text) for each line of the data
val pairs = lines.flatMap(l =>  {
	val label = l.split("\\W+",2).head match{
	case "positive" => 2.0 case "negative" => 0.0 case "neutral" => 1
	}
	val text = l.split("\\W+",2).tail 
  text.map( t => (label, t))
})

//generate dataframe for training data
val training = pairs.toDF("label","text").cache();

//generate dataframe for testing data
val test = linesTest.flatMap(l =>  {
	val label = l.split("\\W+",2).head match{
	case "positive" => 2.0 case "negative" => 0.0 case "neutral" => 1
	}
	val text = l.split("\\W+",2).tail 
  text.map( t => (label, t))
}).toDF("label","text");

 //----------------------------------------- 


/////////////////////////CACH 2 ///////////////////////

//split the raw traing text documents into words, and  add a "words" column to the data frame
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

val wordsTraining = tokenizer.transform(training)

//converts the words column into feature vectors and add to data frame
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
//val hashingTF = new HashingTF().setNumFeatures(20).setInputCol(tokenizer.getOutputCol).setOutputCol("rawFeatures")  
val featurizedTrainingData = hashingTF.transform(wordsTraining)

 
val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
val idfModel = idf.fit(featurizedTrainingData)
val rescaledTrainingData = idfModel.transform(featurizedTrainingData).select("label", "features")
rescaledTrainingData.first

///// processing with testing data
//val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

val wordsTesting = tokenizer.transform(test)

//converts the words column into feature vectors and add to data frame
//val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
val featurizedTestData = hashingTF.transform(wordsTesting)

 
//val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
val idfTestModel = idf.fit(featurizedTestData)
val rescaledTestData = idfTestModel.transform(featurizedTestData).select("label", "features")
rescaledTestData.first





//////////////////FITTING CACH 2///////////////////////////

// Run training algorithm  LR-LBFGS to build the model
val labeledTrain = rescaledTrainingData.map ( r =>  LabeledPoint(r.getDouble(0), r(1).asInstanceOf[Vector] )).cache()
val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(labeledTrain)

//create tuple (label, prediction) for testing set
val predictionAndLabels = rescaledTestData.map ( r =>  LabeledPoint(r.getDouble(0), r(1).asInstanceOf[Vector] ))
  .map{ case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (label, prediction)
}


// create a metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)


// Overall Statistics
val precision = metrics.precision
val recall = metrics.recall 
val f1Score = metrics.fMeasure
println("Summary Statistics")
println(s"Precision = $precision")
println(s"Recall = $recall")
println(s"F1 Score = $f1Score")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}


// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}


///////////////////////// CACH 1
//split the raw text documents into words, and  add a "words" column to the data frame
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

//converts the words column into feature vectors and add to data frame
val hashingTF = new HashingTF().setNumFeatures(20).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
  
  
//train the data using pipeline with LR
val lr = new LogisticRegression().setMaxIter(20).setRegParam(0.01). setNumClasses(3)

//establish pipeline from all stages
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

//val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf))


//////////////////FITTING CACH 1///////////////////////////
// Fit the pipeline to training data
val model = pipeline.fit(training)


// Save the fitted pipeline to disk
model.save("/tmp/lr_model")

// Save the pipeline to disk
pipeline.save("/tmp/pipeline_lr_model")

// load the pipeline for further calculation (if any)
val dupmodel = PipelineModel.load("/tmp/lr_model")


val res=model.transform(test).select($"label",$"text").collect().
foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  } 

sc.stop()

///////////////////////////////////////////////////////


val pos_seq=pos.map(_.split(" ").toSeq)
//c2: val pos_seq = lines.filter(_. startsWith ("positive")).map(line => line.split(" ").toSeq)
pos_seq.first().filterNot(x => x == ',' || x =='.' || x == "positive,")

val encodeLabel = udf[Int, String]( _ match { case "positive" => 1 case "negative" => -1 case "neutral" => 0} )
val toVec4    = udf[Vector, Int, Int, String, String] { (a,b,c,d) => 
  val e3 = c match {
    case "hs-grad" => 0
    case "bachelors" => 1
    case "masters" => 2
  }
  val e4 = d match {case "male" => 0 case "female" => 1}
  Vectors.dense(a, b, e3, e4) 
}
// Prepare training data from a list of (label, features) tuples.
//val sqlContext = new org.apache.spark.sql.SQLContext(sc)

val training = pos.toDF("label","text");

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row

val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)

model.transform(training).select($"title",$"features").map{case Row(label: Double, features: String) => Phrase(label, features)}
   


