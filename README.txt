PREREQUISITES:

1. Kafka and zookeeper must be running with topics "ytcomments", "topic2" created.
2. Elasticsearch, Logstash and Kibana must be installed and RUNNING.
3. Logstash must be started with the given config file as follows.
    $ bin/logstash -f logstash-kafka.conf


HOW TO RUN:

1. Start Kafka as indicated in https://kafka.apache.org/quickstart
2. Create topics "ytcomments" and "topic2"
3. export SPARK_HOME=<path to spark dir>
4. Run below line to start the streaming application
    $ spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_<specific version number> live_comments_analysis.py
5. In new terminal run the kafka producer by:
    $ python data_collector.py

HOW TO TRAIN RNN MODEL: (Expected run time is 8 hours)
execute the RNN_Training.ipynb 


make sure packages argument matches your Spark and Scala 
environment. You can check Spark and Scala versions by seeing 
them on spark-shell
