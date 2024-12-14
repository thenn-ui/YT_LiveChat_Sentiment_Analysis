from __future__ import print_function

import sys
import os

#sys.path.append("/home/thenn/spark/spark-3.5.1-bin-hadoop3/bin")

sys.path.append(os.environ['SPARK_HOME'] + "bin")

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

from pyspark.sql.functions import concat_ws,col,udf
from pyspark.sql.types import StringType, ArrayType

import tensorflow as tf

#def testfunc():

from rnnmodel import *

#define the UDF for prediction
getSentimentUDF = udf(lambda x:get_sentiment(x, model, tokenizer),ArrayType(StringType())) 

if __name__ == "__main__":
    
    model = tf.keras.models.load_model("sentiment_analysis_model.h5")
    tokenizer = None
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    bootstrapServers = "localhost:9092" 
    subscribeType =  "subscribe" 
    topics =  "ytcomments" 

    spark = SparkSession\
        .builder\
        .appName("StructuredKafkaNamedEntityCount")\
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Create DataSet representing the stream of input lines from kafka
    lines = spark\
        .readStream\
        .format("kafka")\
        .option("kafka.bootstrap.servers", bootstrapServers)\
        .option(subscribeType, topics)\
        .load()\
        .selectExpr("CAST(value AS STRING)")


    # Split the lines into words
    predictiondf = lines.toDF("value").select(
            getSentimentUDF(col("value")).alias("value")
    )

    sentiments = predictiondf.select(predictiondf.value[0].alias("input"), predictiondf.value[1].alias("sentiment"))

    sentiments = sentiments.select(concat_ws('=', "input", "sentiment").alias("value"))

    # Start running the query that prints the running counts to the console
    query = sentiments\
        .writeStream\
        .outputMode('append')\
        .format('console')\
        .trigger(processingTime='2 seconds')\
        .start()
    

    '''publisher = concatdf.selectExpr("CAST(value AS STRING)")\
        .writeStream\
        .outputMode('complete')\
        .format('kafka')\
        .option("kafka.bootstrap.servers", bootstrapServers)\
        .option("publish", "topic2")\
        .option("topic", "topic2")\
        .option("checkpointLocation", "./checkpoints/")\
        .start()
    '''
    query.awaitTermination()
    #publisher.awaitTermination()
