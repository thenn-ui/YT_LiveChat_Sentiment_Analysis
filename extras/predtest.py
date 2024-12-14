from rnnmodel import *

model = tf.keras.models.load_model("sentiment_analysis_model.h5")
tokenizer = None
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

print(get_sentiment("Hello Hawaii! Aloha to everyone", model,tokenizer))

model.summary()