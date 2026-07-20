# import libraries 
import streamlit as st
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# read a csv
df=pd.read_csv("house_prices_practice.csv")

# Feature X and Target y
X=df[[
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "LotArea"
]]

y=df["SalePrice"]

# train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)

# create a model
model=LinearRegression()

# train the model
model.fit(X_train,y_train)

# save model
# with:- safely handle file
# open: open file
# "model.pkl":- file name
# wb:- write binary mode
# as:- give variable name
# f:- file object
# : start block
# pickle:- python saving module
# dump:- save object
# model:- trained Ml model
# f:- file where model is saved
with open("model.pkl","wb") as f: 
    pickle.dump(model,f)

# load model
# rb:- read binary mode
# loaded_model:- variable storing loaded model
# load:- load/read object
with open("model.pkl","rb") as f:
    loaded_model=pickle.load(f)

# streamlit ui
# title
st.title("House Price Prediction App")

st.write("Enter House details")

# User Input
OverallQual=st.number_input("Overall Quality")

GrLivArea=st.number_input("Ground Living Area")

GarageCars=st.number_input("Garage Cars")

TotalBsmtSF=st.number_input("Total Basement Square Feet")

YearBuilt=st.number_input("Year Built")

FullBath=st.number_input("Full Bathrooms")

BedroomAbvGr=st.number_input("Bedrooms Above Ground")

LotArea=st.number_input("Lot Area")

# prediction
if st.button("Predict House Price"):

    # create a DataFrame
    new_data=pd.DataFrame([[
        OverallQual,
        GrLivArea,
        GarageCars,
        TotalBsmtSF,
        YearBuilt,
        FullBath,
        BedroomAbvGr,
        LotArea
    ]], columns=[
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "LotArea"
    ])

    # predict
    prediction=loaded_model.predict(new_data)

    # display result
    st.success(f"Predicted House Price: {prediction[0]:,.2f}")