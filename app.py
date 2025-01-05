import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime

model=load_model(r'C:\Users\HP\Desktop\stock\Stock price prediction.keras')

st.header('stock market predictor')
stock=st.text_input('Enter stock symbol','GOOG')
start='2012-01-01'
end=datetime.now()
data=yf.download(stock,start,end)
st.subheader('Stock data')
st.write(data)

train_data=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
test_data=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

past_100_days_data=train_data.tail(100)
test_data=pd.concat([past_100_days_data,test_data],ignore_index=True)

scaled_test_data=scaler.fit_transform(test_data)
st.subheader('Price vs MA-50')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'b')
plt.plot(data.Close,'g')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA-50 vs MA-100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'b')
plt.plot(data.Close,'g')
plt.plot(ma_100_days,'r')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA-100 vs MA-200')
ma_200_days=data.Close.rolling(200).mean()
fig3=plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')
plt.plot(ma_200_days,'y')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100,scaled_test_data.shape[0]):
    x.append(scaled_test_data[i-100:i])
    y.append(scaled_test_data[i,0])
x,y=np.array(x),np.array(y)
predict=model.predict(x)

scale=1/scaler.scale_

predict=predict*scale
y=y*scale

st.subheader('Original Price vs Predicted Price')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'b',label='predicted price')
plt.plot(y,'g',label='original price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
st.pyplot(fig4)