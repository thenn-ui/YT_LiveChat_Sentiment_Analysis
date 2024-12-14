from comments import get_video_comments
from kafkaproducer import *
import pytchat

video_id = input("Enter the video Id: ") # video Id can be found in the youtube url after the ?v= parameter
# example: https://www.youtube.com/watch?v=WHHmIfog0Fs | video id = WHHmIfog0Fs

type = int(input("Enter 1 for comments and 2 for live chat: "))

COMMENTS = 1
LIVECHAT = 2

data = []

init_kafka_producer("localhost:9092")

if type == COMMENTS: # 
    print(video_id)
    data = get_video_comments(video_id)
    print(data)
    punch_to_kafka(data, "ytcomments")

elif type == LIVECHAT:
    chat = pytchat.create(video_id)

    try:
        while chat.is_alive():
            for c in chat.get().sync_items():
                print(c.message)
                punch_to_kafka(c.message, "ytcomments")   
                

    except Exception as e:
        # TODO: Parse error logs
        print(e)
        print(f"Exception occured with the payload: {c.message}")
        exit()
else:
    exit()

