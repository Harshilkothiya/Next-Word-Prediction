import streamlit as st
import numpy as np
import time
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")

print("start this file")

model = None
tokenizer = None

def load_models():
    global model
    with open("../Models/model.pkl", "rb") as f:
        model = joblib.load(f)

    global tokenizer
    with open("../Models/tokenizer.pkl",'rb') as f:
        tokenizer = joblib.load(f)

def predict(text):
    token = tokenizer.texts_to_sequences([text])[0]
    token = pad_sequences([token], maxlen = 50, padding='pre')
    output = np.argmax(model.predict(token))

    for word, index in tokenizer.word_index.items():
        if index == output:
            return text + " " + word
    

# Home page
st.set_page_config(
    page_title="Smart Crop",
    page_icon="logo.webp",
    layout="centered",
)

def main():
    html_text = """
    <style>
    p {font-size :17px;}
    .block-container {padding: 2rem 1rem 3rem;}
    MainMenu {visibility: hidden;}
    </style>
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;">Next Word Prediction: Game of Throne</h1>
    </div>
    """
    st.markdown(html_text, unsafe_allow_html=True)
    st.subheader("Find the next word of your word")
    word = st.text_input("Enter the word")
    number = st.number_input("Enter the How many number you want", 1)

    if st.button('Predict'):
        placeholder = st.empty()
        for i in range(number):
            word = predict(word)
            placeholder.success(''.join(word))


if __name__ == "__main__":
    load_models()
    main()