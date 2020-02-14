package ca.uwaterloo.cs451

//import io.bespin.scala.util.Tokenizer

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
//import sqlContext.implicits._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql
import org.apache.spark.sql.functions.lower
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.ml.feature.IDF

class Conf2(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, reducers)
  //mainOptions = Seq(input,output)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = false)
  val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}
 
//object p1 extends Tokenizer {
  object p2 {
  val log = Logger.getLogger(getClass().getName())
case class rev(review:String, label:Int)
  def main(argv: Array[String]) {
    val args = new Conf2(argv)

    log.info("Input: " + args.input())
   // log.info("Output: " + args.output())
    //log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("p2")
    val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
   // val outputDir = new Path(args.output())
   // FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)
val spark = org.apache.spark.sql.SparkSession.builder
        .master("local")
        .appName("Spark CSV Reader")
        .getOrCreate;

    //reading csv file and converting it into dataframe
//     val df = sqlContext.read
//     .format("com.databricks.spark.csv")
//     .option("header", "true") // Use first line of all files as header
//     .option("inferSchema", "true") // Automatically infer data types
//     .option("quote", "\"")  //escape the quotes 
//     .option("ignoreLeadingWhiteSpace", true)  // escape space before your data
//     .load(args.input())
    
    val df = spark.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load(args.input())
      .toDF()
    
    print(s"o1->,$df")
    
    //data manipulation
    val df1=df.select("Text","Score")
//      df1.map(row =>{ 
//        val r1=row.getAs[Int](1)
//        val r2= if (r1>=3) 1 else 0
//                     Row(row(0),r2)  
//      }).collect()
   // df1.withColumn("score", $"score".cast(sql.types.IntegerType))
   // df1.withColumn("target", 'score.cast("Int")).select('target as 'text, 'target)
   val df2= df1.withColumn("Score",col("Score").cast("Integer"))
    //val df1=df.withColumn("score", when($"score" >= 3, 1))
    print(s"o2->,$df1")
    print(s"o3->,$df2")
    
    df2.show()
//      df2.map(row =>{ 
//         val r1=row.getAs[Int](1)
//         val r2= if (r1>=3) 1 else 0
//              Row(row(0),r2)  
//       }).collect()
   val df3=df2.withColumn("Score", when($"Score">=3,1).otherwise(0))
       df3.show()
   df3.groupBy("Score").count.show()
    //val df4=df3.select($"Text", lower($"Text"))
    
    //converting text to lowercase
    val df4=df3.withColumn("Text", lower(col("Text")));
    df4.show()
    
 // val Array(training_data, test_data)=df4.randomSplit(Array(0.7,0.3))
    
   // training_data.show()
    //tokenization
    val tokenizer = new Tokenizer()
                     .setInputCol("Text")
                     .setOutputCol("Words")
    val tokenizedData = tokenizer.transform(df4)
    //val tokenizedData = tokenizer.transform(training_data)
    
    //stopword removal
    val remover = new StopWordsRemover()
                  .setInputCol("Words")
                  .setOutputCol("filtered_words")

//     val dataSet = spark.createDataFrame(Seq((0, Seq("I", "saw", "the", "red", "balloon")),
//   (1, Seq("Mary", "had", "a", "little", "lamb"))
// )).toDF("id", "raw")

   val r= remover.transform(tokenizedData)
    
    val stemmed_data = new Stemmer()
                .setInputCol("filtered_words")
                .setOutputCol("stemmed_words")
                .setLanguage("English")
                .transform(r)

   stemmed_data.show()
    val df5=stemmed_data.select("stemmed_words","Score")
    //val df6=df5.withColumnRenamed("stemmed_words", "Text")
    val df6=df5.withColumn("Text", concat_ws(" ",$"stemmed_words"))
    val df7=df6.select("Text","Score")
    val df8=df7.withColumnRenamed("Score", "label")
    //df5.show()
    //df6.show()
   // df7.show()
    df8.show()
   //df8.filter("label is null").show
    //println(s"$df8")
    val df9=df8.withColumnRenamed("Text", "text")
   // val df9=df4.withColumnRenamed("Score", "label")
    df9.show()
    
   //val df0=df9.filter
    //TF-IDF
    val tokenizer1 = new Tokenizer()
                     .setInputCol("text")
                     .setOutputCol("words")
                     //.transform(df9.na.fill(Map("text" -> "")))
   // val t= tokenizer1.transform(df9) 
   // val t=tokenizer1.transform(df9.na.fill(Map("text" -> "")))
    
    val hashingTF = new HashingTF()
                     .setNumFeatures(1000)
                     .setInputCol(tokenizer1.getOutputCol)
                     .setOutputCol("hashing")
                    // .setOutputCol("features")
   // val idfModel = idf.fit(hashedData)
   // val hdata=hashingTF.transform(t)
    //val hashedData = hashingTF.transform(t)
    
   val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
   // val idfModel = idf.fit(hashedData)
    
    //idfModel.show()
    
    //model evaluation
     val LR=new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.01)
      val pipeline=new Pipeline()
            //  .setStages(Array(tokenizer1,hashingTF,LR))
                   .setStages(Array(tokenizer1,hashingTF,idf,LR))
    
 // val model=pipeline.fit(df9)
//    training model
//     val pipeModel=pipeline.fit(training_data)
//     val trainPred=pipeModel.transform(training_data)
//     val testPred=pipeModel.transform(test_data)
//    println(s"$trainPred")
    
     val eval=new BinaryClassificationEvaluator()
    val pg=new ParamGridBuilder()
          //.addGrid(hashingTF.numFeatures,Array(10000,100000))
          .addGrid(LR.regParam,Array(0.01,0.1,0.001,1.0,100,200))
          //.addGrid(LR.maxIter,Array(20,30))
          .build()
    
    val CV=new CrossValidator()
              .setEstimator(pipeline)
              .setEstimatorParamMaps(pg)
              .setNumFolds(10)
              .setEvaluator(eval)
    val CVModel=CV.fit(df9)
    val avgF1Scores = CVModel.avgMetrics
    //println(avgF1Scores.mkString(" "))
   println(CVModel.avgMetrics.toList)
   // sc.parallelize(avgF1Scores).saveAsTextFile(args.output())
   
    
  }
    
  }
