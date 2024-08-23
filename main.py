import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imbd.h5')

##Helper Function - 
def pre_process_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

##Streamlit Application
st.title("IMDB Review Sentiment Analysis")
st.write("Enter a movie review that will classifiy as postive or Negative")

#User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessedText = pre_process_text(user_input)

    prediction = model.predict(preprocessedText)
    sentiment = 'Postive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write("Please Enter a Moview Review")    