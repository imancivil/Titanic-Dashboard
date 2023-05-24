import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')
st.title('Would you have survived if you were among the passengers on the Titanic? :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
passengerid = st.text_input("Input Passenger ID", '175') 
pclass = st.selectbox("Choose class", [1,2,3])
name  = st.text_input("Input Passenger Name", '')
sex = st.select_slider("Choose sex", ['Male','Female'])
age = st.slider("Choose age",0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,10)
ticket = st.text_input("Input Ticket Number", "",0,1000) 
fare = st.number_input("Input Fare Price", 0,600)
cabin = st.text_input("Input Cabin", "example: C123") 
# embarked = st.select_slider("Did they Embark?", ['S','C','Q'])
embarked = st.selectbox("The port of embarkation? (C=Cherbourg, Q=Queenstown, S=Southampton)", ['S', 'C', 'Q'])

def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)

