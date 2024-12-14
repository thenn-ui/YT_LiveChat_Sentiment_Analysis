#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install tensorflow


# In[ ]:

import sys
import os

#sys.path.append("/home/thenn/spark/spark-3.5.1-bin-hadoop3/bin")

sys.path.append(os.environ['SPARK_HOME'] + "bin")


import pyspark
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

# In[ ]:
conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

'''twitterData = spark.read.option("header","false").option("inferSchema","true").csv("dbfs:/FileStore/tweets.csv").toDF("label","userID","date","flag","user","text").select("text", "label")

twitterData = twitterData.withColumn('label', F.when(col("label") == 4, 1).otherwise(col("label")))
twitterData = twitterData.withColumn('label', F.when(col("label") == 0, -1).otherwise(col("label")))

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
twitterWordsData = tokenizer.transform(twitterData)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
twitterFilteredData = remover.transform(twitterWordsData)

twitterFinalData = twitterFilteredData.select("label", "filtered")
twitterFinalData = twitterFinalData.filter(col("filtered")[0].isNotNull())

# Split the data into training, validation, and testing sets
(twitterTrainingData, twitterValidationData, twitterTestingData) = twitterFinalData.randomSplit([0.7, 0.1, 0.2], seed=1234)

twitter_train_label = twitterTrainingData.select("label")
twitter_train_data = twitterTrainingData.select("filtered")

twitter_valid_label = twitterValidationData.select("label")
twitter_valid_data = twitterValidationData.select("filtered")

twitter_test_label = twitterTestingData.select("label")
twitter_test_data = twitterTestingData.select("filtered")'''


# In[ ]:


# Reading the Data
rtData = spark.read.option("header","true").option("inferSchema","true").csv("rt_reviews.csv").toDF("label","text").select("text","label").limit(10000)

# Modifying the Labels
rtData = rtData.withColumn('label', F.when(col("label") == "fresh", 1).otherwise(col("label")))
rtData = rtData.withColumn('label', F.when(col("label") == "rotten", -1).otherwise(col("label")))

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
rtWordsData = tokenizer.transform(rtData)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
rtFilteredData = remover.transform(rtWordsData).select("label", "filtered")
rtFilteredData = spark.createDataFrame(rtFilteredData.rdd.map(lambda x: (x[0], [word for word in x[1] if len(word) > 0])).collect(), ['label','filtered'])

# Run StringIndexer on all the words in the data
allWords = spark.createDataFrame(rtFilteredData.select("filtered").rdd.flatMap(lambda x: x[0]), StringType())
indexer = StringIndexer(inputCol = "value", outputCol="categoryIndex")
indexed = indexer.fit(allWords).transform(allWords)

# Create a dictionary with the word as key and the index as value
index_dict = dict(indexed.distinct().rdd.map(lambda x: (x["value"], x["categoryIndex"])).collect())

# Convert the columns with list of words to list of idexes, using the index_dict
rtFinalData = spark.createDataFrame(rtFilteredData.rdd.map(lambda x: (x[0], x[1], [index_dict[word] for word in x[1] if len(word) > 0])).collect(), ['label','filtered', 'filteredIndex'])
rtFinalData = rtFinalData.select("label", "filteredIndex")


# In[ ]:


# Split the data into training and testing sets
(rtTrainingData, rtValidationData, rtTestingData) = rtFinalData.randomSplit([0.7, 0.1, 0.2], seed=1234)

rt_train_label = rtTrainingData.select("label")
rt_train_data = rtTrainingData.select("filteredIndex")

rt_valid_label = rtValidationData.select("label")
rt_valid_data = rtValidationData.select("filteredIndex")

rt_test_label = rtTestingData.select("label")
rt_test_data = rtTestingData.select("filteredIndex")


# In[ ]:


# Convert the DataFrames to lists of arrays
rt_train_data = rt_train_data.rdd.flatMap(lambda x: np.array(x)).collect()
rt_train_label = np.array(rt_train_label.rdd.flatMap(lambda x: x).collect())

rt_valid_data = rt_valid_data.rdd.flatMap(lambda x: np.array(x)).collect()
rt_valid_label = np.array(rt_valid_label.rdd.flatMap(lambda x: x).collect())

rt_test_data = rt_test_data.rdd.flatMap(lambda x: np.array(x)).collect()
rt_test_label = np.array(rt_test_label.rdd.flatMap(lambda x: x).collect())


# In[ ]:


# Pad the data, so each column is of the same length
rt_train_data = sequence.pad_sequences(rt_train_data, maxlen = 40)
rt_valid_data = sequence.pad_sequences(rt_valid_data, maxlen = 40)
rt_test_data = sequence.pad_sequences(rt_test_data, maxlen = 40)


# In[ ]:


rnn_model1 = Sequential()
rnn_model1.add(Embedding(1000, 64))
rnn_model1.add(SimpleRNN(units = 32, dropout = 0.2, recurrent_dropout = 0.2))
rnn_model1.add(Dense(1, activation = 'sigmoid'))

rnn_model1.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

history_rnn1 = rnn_model1.fit(rt_train_data, rt_train_label.astype(int), batch_size = 32, epochs = 10, validation_data = (rt_valid_data, rt_valid_label.astype(int)))


# In[ ]:


rnn_model.evaluate(rt_test_data, rt_test_label.astype(int), verbose=0)


# In[ ]:


rnn_model2 = Sequential(name="Simple_RNN")
rnn_model2.add(Embedding(10000, 64))
rnn_model2.add(SimpleRNN(128, activation='tanh', return_sequences=True))
rnn_model2.add(SimpleRNN(64, activation='tanh', return_sequences=False))
rnn_model2.add(Dense(1, activation='sigmoid'))

rnn_model2.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

history_rnn2 = rnn_model2.fit(rt_train_data, rt_train_label.astype(int), batch_size=64, epochs=10, verbose=1, validation_data = (rt_valid_data, rt_valid_label.astype(int)))


# In[ ]:


rnn_model2.evaluate(rt_test_data, rt_test_label.astype(int), verbose=0)

