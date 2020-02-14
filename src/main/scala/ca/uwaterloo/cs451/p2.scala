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
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.functions.concat_ws

class Conf3(args: Seq[String]) extends ScallopConf(args) {
  //mainOptions = Seq(input, output, reducers)
  mainOptions = Seq(input)
  val input = opt[String](descr = "input path", required = true)
  //val output = opt[String](descr = "output path", required = true)
  //val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}
 
//object p1 extends Tokenizer {
  object p3 {
  val log = Logger.getLogger(getClass().getName())
case class rev(review:String, label:Int)
  def main(argv: Array[String]) {
    val args = new Conf3(argv)

    log.info("Input: " + args.input())
   // log.info("Output: " + args.output())
    //log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("p3")
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
    df.show()
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
    
 // val Array(training_data, test_data)=df4.randomSplit(Array(0.5,0.5))
    
   // training_data.show()
    //tokenization
    val tokenizer = new Tokenizer()
                     .setInputCol("Text")
                     .setOutputCol("Words")
    val tokenizedData = tokenizer.transform(df4)
    
    //stopword removal
    val remover = new StopWordsRemover()
                  .setInputCol("Words")
                  .setOutputCol("filtered_words")

//     val dataSet = spark.createDataFrame(Seq((0, Seq("I", "saw", "the", "red", "balloon")),
//   (1, Seq("Mary", "had", "a", "little", "lamb"))
// )).toDF("id", "raw")

   val r= remover.transform(tokenizedData)
    
    r.show()
    
//     val stemmed_data = new Stemmer()
//                 .setInputCol("filtered_words")
//                 .setOutputCol("stemmed_words")
//                 .setLanguage("English")
//                 .transform(r)

//    stemmed_data.show()
    val df5=r.select("filtered_words","Score")
    //val df6=df5.withColumnRenamed("stemmed_words", "Text")
    val df6=df5.withColumnRenamed("filtered_words", "text")
    //val df6=df5.withColumn("text", concat_ws(" ",$"filtered_words"))
    //val df7=df6.select("text","Score")
    val df7=df6.withColumnRenamed("Score", "label")
    //df5.show()
    //df6.show()
   // df7.show()
    df7.show()
    
    val df8=df7.filter($"label"===1)
    
    df8.show()
   //df8.filter("label is null").show
    //println(s"$df8")
   // val df9=df8.withColumnRenamed("Text", "text")
   // val df9=df4.withColumnRenamed("Score", "label")
    //df9.show()
    val bg=new NGram().setInputCol("text").setOutputCol("bigram").setN(2);
    val bgdf=bg.transform(df8);
    
    bgdf.show();
    
    val rs=bgdf.select("bigram")
    rs.show()
    val rs1=rs.rdd.map(p=>p(0).toString)
//     //val rs1=rs.rdd.map(p=>p.getAs[Seq[Row]])
//              val rs2=  .select(concat_ws(",", $"arrayCol"))
    rs1.take(2)
       .foreach(println)
   
    
   // rs1.select($"name",flatten($"subjects")).show(false)
    
      val rs2=rs1
      //val rs2=rs.rdd
//     .flatMap{case p => for {x <- p} yield x}
    .flatMap(p =>{
       p.split(",")
             .map(p => p.mkString("")).toList
      })
  // .filter(p=>(!p.contains("/><")).filter(!p.contains(">br")).filter(!p.contains("/>")).filter(!p.contains("<")))
    .filter(p=>(!p.contains("/><"))).filter(p=>(!p.contains("br"))).filter(p=>(!p.contains("<"))).filter(p=>(!p.contains("/>")))
                                                                                                      
   .map(p=>p.replaceFirst("^\\s*",""))
    .map(p=>p.replaceFirst("[^A-Za-z0-9 ]*","")) 
   // .map(p=>p.replaceFirst("^\\-*",""))
   .map(p=>p.replaceAll("""(?m)\s*$""","")) 
    .map(p=>p.replaceAll("""(?m)[^A-Za-z0-9 ]*$""","")) 
    //.map(p=>p.replaceAll("""(?m)\.*$""","")) 
//     .map(p=>p.replaceAll("""(?m)\-*$""",""))
//      .map(p=>p.replaceAll("""(?m)\)*$""",""))
//     .map(p=>p.replaceAll("""(?m)\.*$""",""))
//     .map(p=>p.replaceAll("""(?m)\s*$""","")) 
//     .filter(p=>!p.equals(""))
    .filter(p=>p.contains(" "))
   // .filter(p=>p.contains("WrappedArray"))
   .map(p=>p.replaceAll("\\s*\\bWrappedArray\\(\\b\\s*","")) 
    .map(p=>p.replaceFirst("^\\s*",""))
    .map(p=>p.replaceAll("""(?m)\s*$""","")) 
    .filter(p=>p.contains(" "))
   // .filter(p=> _.count(' '==) >= 3)
  //  .map(p=> )
   // .foreach(println)
   // .map(p=>(p(0).replace(" ","_"),p(1)))
     .map(p=>(p,1))
     .reduceByKey(_+_)
   // .map(p=>(p(0).replace(" ","_")))
     .sortBy(-_._2)
    
   rs2.take(100)
       .foreach(println)
    
    
  }
    
  }
