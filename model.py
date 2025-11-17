
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import warnings
import fbprophet

#Data were we have to work
from google.colab import files
uploaded = files.upload()
data = pd.read_csv("GOOG.csv")
data.head()


#Before moving forward, let’s visualize the data so that we could get some better insights into the data we will work on:
plt.style.use("fivethirtyeight")
plt.figure(figsize=(16,8))
plt.title("Google Closing Stock Price")
plt.plot(data["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

#Only two features are needed from the dataset that is Date and Close Prices. So let’s prepare the data for our model:

data = data[["Date","Close"]] 
data = data.rename(columns = {"Date":"ds","Close":"y"})
data.head()


#Now let’s fit the data to the Facebook Prophet model for stock price prediction of Google:

from fbprophet import Prophet
m = Prophet(daily_seasonality=True)
m.fit(data)

#We have successfully fit the data to the Facebook Prophet model. Now let’s have a look at the stock price prediction made by the model:
future = m.make_future_dataframe(periods=365)
predictions=m.predict(future)

#Now let’s have a look at the seasonal affects on this prediction that is made by our model:
m.plot(predictions)
plt.title("Prediction of GOOGLE Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Stock Price")
plt.show()

