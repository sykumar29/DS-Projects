#Description: This program detects if someone has diabetes using machine learning and python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create a title and subtitle

st.header("""
Diabetes Detection Using ML
""")

#Open and display an image

image=Image.open('C:/Users/sykum/Documents/Fun Projects/ML/WebApp/diabetes-head.jpg')
st.image(image,use_column_width=True)

#Get the data
df=pd.read_csv('C:/Users/sykum/Documents/Fun Projects/ML/WebApp/diabetes.csv')

#Set the subheader
st.subheader('Data Information:')

#Show the data as a table
st.dataframe(df)

#Show statistics on the data
st.write(df.describe())

#Show the data as a chart
chart=st.bar_chart(df)

#Split the data into independent x and dependent y
x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values

#Split the data set into 75% training and 25% testing
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#Get the feature input from the user
def get_user_input():
    pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    glucose=st.sidebar.slider('Glucose',0,199,117)
    bloodpressure=st.sidebar.slider('Blood Pressure',0,122,72)
    skinthickness=st.sidebar.slider('Skin Thickness',0,99,23)
    insulin=st.sidebar.slider('Insulin',0.0,846.0,30.0)
    bmi=st.sidebar.slider('BMI',0.0,67.1,32.0)
    dpf=st.sidebar.slider('Diabetes Pedigree Function',0.078,2.42,0.3725)    
    age=st.sidebar.slider('Age',21,81,29)


    #Store a dictionary into a variable
    user_data={'Pregnancies':pregnancies,
               'Glucose':glucose,
               'Blood Pressure':bloodpressure,
               'Skin Thickness':skinthickness,
               'Insulin':insulin,
               'BMI':bmi,
               'DPF':dpf,
               'Age':age
               }

    #Transform the data into a dataframe
    features=pd.DataFrame(user_data,index=[0])
    return features



#Store the user input to a variable
user_input=get_user_input()

#Set a subheader and display the user's input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(xtrain,ytrain)

#Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(ytest,RandomForestClassifier.predict(xtest))*100)+'%')

#Store the model's prediction in a variable
prediction=RandomForestClassifier.predict(user_input)

#Set a subheader and display the classification
st.subheader('Prediction:')
if prediction[0]==1:
    st.write('You are Diabetic.')
else:
    st.write('You are not Diabetic.')





