import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
 
df = pd.read_csv('sentiment_review.csv')
x = df.iloc[:,0].values # Review column as input
y = df.iloc[:,1].values # Sentiment column as output
st.title("Sentiment Analysis On Review Datasets")
st.subheader('TFIDF Vectorizer')
st.write('This project is based on Naive Bayes Classifier')
 
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x,y)
message = st.text_area("Enter Text","Type Here ..")
op = text_model.predict([message])
if st.button("Predict"):
  st.title(op)
