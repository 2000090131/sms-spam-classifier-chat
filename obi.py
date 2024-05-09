import streamlit as st
import pickle
import string
import pandas as pd
import numpy as np
import base64
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to preprocess and classify SMS
def classify_sms(input_sms):
    ps = PorterStemmer()

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]

    return result

# Streamlit UI
st.title("SMS Spam Classifier Chat")

# Sidebar with Introduction and About sections
st.sidebar.title("Sections")

# Add GIF above "Sections" title
gif_url = "https://raw.githubusercontent.com/2000090131/2000090131/main/wired-lineal-1203-bear.gif"  # Replace with your GIF URL
gif_html = f'<div style="display: flex; justify-content: center;"><img src="{gif_url}" style="width: 60%; margin-top: -20px;"></div>'

st.sidebar.markdown(gif_html, unsafe_allow_html=True)

selected_section = st.sidebar.radio("", ["Introduction", "About"])

# Add introduction section
if selected_section == "Introduction":
    st.sidebar.header("Introduction")
    intro_text = """ 
    The SMS Spam Classifier project employs machine learning and natural language processing techniques 
        to address the common issue of unwanted SMS messages. Its core objective is to create a versatile 
        and efficient classifier for rapid spam detection. The project utilizes feature engineering, 
        incorporating TF-IDF and word embeddings, and explores a wide range of machine learning algorithms 
        and deep learning models to achieve this goal.   
    """
    st.sidebar.write(intro_text)

# Add about section
elif selected_section == "About":
    st.sidebar.header("About")
    about_text = """ 
    2000090131 , 
    2000090044 ,
    2000090050 ,
    2000090112
    """
    st.sidebar.write(about_text)
    developers = [
        {
            "name": "Manas Ranjan",
            "role": "Team Leader",
            "image_url": "https://raw.githubusercontent.com/2000090131/2000090131/main/team%20member/131.jpg",
        },
        {
            "name": "Manikanta",
            "role": "Team Member",
            "image_url": "https://raw.githubusercontent.com/2000090131/2000090131/main/team%20member/44.png",
        },
        {
            "name": "Hari Surya",
            "role": "Team Member",
            "image_url": "https://raw.githubusercontent.com/2000090131/2000090131/main/team%20member/50.png",
        },
        {
            "name": "Shiva Krishna Teja",
            "role": "Team Member",
            "image_url": "https://raw.githubusercontent.com/2000090131/2000090131/main/team%20member/112.png"
        },
    ]
    # Create the About Us page
    st.title("About Us")
    # Create a row for each developer with two photos side by side
    for i in range(0, len(developers), 2):
        col1, col2 = st.columns(2)
        with col1:
            st.image(developers[i]["image_url"], width=120)
            st.subheader(developers[i]["name"])
            st.write(developers[i]["role"])

        if i + 1 < len(developers):
            with col2:
                st.image(developers[i + 1]["image_url"], width=120)
                st.subheader(developers[i + 1]["name"])
                st.write(developers[i + 1]["role"])

        st.write("")  # Add a space between rows
    st.write("")  # Add a space between rows

# Main content on the right side
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt:= st.chat_input("Enter a message:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Classify SMS
    result = classify_sms(prompt)

    # Display spam/ham prediction in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if result == 1:
            assistant_response = "This looks like spam!"
        else:
            assistant_response = "This seems to be a legitimate message."

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
