import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

nav = st.sidebar.radio("Navigation", ["About", "Predict"])
df = pd.read_csv("insurance.csv")
df.replace({'sex': {'male': 0, 'female': '1'}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': '1'}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
image_path = "https://www.iii.org/sites/default/files/p_lifeinsurance12_821759032.jpg"
X = df.drop(columns='charges', axis=1)
y = df["charges"]

rfr = RandomForestRegressor()
rfr.fit(X, y)
if nav == "About":
    st.title("Insurance Premium Predictor")
    st.write("Introducing my cutting-edge Insurance Premium Predictor application, meticulously developed using the powerful RandomForestRegressor model from scikit-learn in Python. This app offers users an opportunity to input essential personal information, including age, sex, region, BMI, number of children, and smoking status. By leveraging the prowess of advanced machine learning algorithms, this app generates an accurate and tailored insurance premium estimate based on the provided details. Experience the convenience of obtaining a precise and personalized insurance premium prediction like never before with this intelligently designed and user-friendly application.")
    st.text("")
    st.image(image_path,width=700)
    st.text("")
    st.text(" ")
    st.subheader("Created By : Rhythm Suthar")
if nav == "Predict":
    st.title("Enter details")
    age = st.number_input("Age:", step=1, min_value=1)
    sex = st.radio("Sex", ("Male", "Female"))

    if sex == "Male":
        s = 0
    if sex == "Female":
        s = 1
    bmi = st.number_input("BMI", min_value=0)
    children = st.number_input("Enter number of children:", min_value=0)

    smoke = st.radio("Do you smoke?", ("Yes", "No"))
    if smoke == "Yes":
        sm = 0
    if smoke == "No":
        sm = 1

    region = st.selectbox('Region', ('SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'))

    if region == "SouthEast":
        reg = 0
    if region == "SouthWest":
        reg = 1
    if region == "NorthEast":
        reg = 2
    if region == "NorthWest":
        reg = 3

    if st.button("Predict"):
        st.subheader("Predicted Premium")
        res =st.text(rfr.predict([[age, s, bmi, children, sm, reg]]))
        st.subheader("Accuracy:")
        st.write(rfr.score(X,y)*100)
