from kafka import KafkaProducer
import time


def init_kafka_producer(bootstrap_server):
    global producer 
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

def punch_to_kafka(data, topic):

    if isinstance(data, list):
        for record in data:
            producer.send(topic, record.encode("UTF-8"))

        producer.flush()

    elif isinstance(data, str):
        producer.send(topic, data.encode("UTF-8"))
        producer.flush()
        

    else:
        print("Error")