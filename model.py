import pandas as pd
from prophet import Prophet   # Updated from fbprophet → prophet

def train_model():
    # Load your dataset
    data = pd.read_csv("GOOG.csv")

    # Keep only needed columns
    data = data[["Date", "Close"]]
    data = data.rename(columns={"Date": "ds", "Close": "y"})

    # Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    return model


def make_prediction(model, days=30):
    # Create future dataframe
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Return only last N days predictions
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return result
