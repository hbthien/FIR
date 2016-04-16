

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.feature.RegexTokenizer

case class WikiData(category: String, text: String)
case class LabeledData(category: String, text: String, label: Double)

val wikiData = sc.parallelize(List(WikiData("Spark", "this is about spark"), 
    WikiData("Hadoop","then there is hadoop"), WikiData("Scala","a programming language"),
    WikiData("Hadoop","greagea fdajk fadkljlj fda    l")))

val categoryMap = wikiData.map(x=>x.category).distinct.zipWithIndex.mapValues(x=>x.toDouble).collectAsMap

val labeledData = wikiData.map(x=>LabeledData(x.category, x.text, categoryMap.get(x.category).getOrElse(0.0))).
  toDF("category", "text", "label")

val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("/W+")
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

val model = pipeline.fit(labeledData)

model.transform(labeledData).show