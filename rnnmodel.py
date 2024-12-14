import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle



def init_models():
    # Load the pre-trained model
    global model
    model = tf.keras.models.load_model("sentiment_analysis_model.h5")
    print("[========= Loaded RNN Model =========]")

    global tokenizer
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("[========= Loaded Tokenizer =========]")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text == "":
        return None
    else:
        return text.strip()


# Function to preprocess the text input
def preprocess_text(text, tokenizer):
    # Clean the text
    text = clean_text(text)
    if text is None:
        return None
    # Calculate max sequence length based on the number of words in the input text
    max_len = len(text.split())
    # Convert the text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Function to interpret prediction
def interpret_prediction(prediction):
    if prediction < 0.4:
        return 'Negative'
    elif 0.4 <= prediction <= 0.7:
        return 'Neutral'
    else:
        return 'Positive'
    

def get_sentiment(input_text, model, tokenizer):

    processed_input = preprocess_text(input_text, tokenizer)
    res = []
    res.append(input_text)

    if processed_input is None:
        res.append("NA")
        return res
    
    print(f"Input text: {input_text}")
    print(f"Processed input: {processed_input} {len(processed_input)}")

    prediction = model.predict(processed_input)
    sentiment = interpret_prediction(prediction[0][0])
    
    print(f"Sentiment: {sentiment}\n")
    res.append(sentiment)
    return res
