from kafkaproducer import *

init_kafka_producer("localhost:9092")

punch_to_kafka("Hello world", "ytcomments")