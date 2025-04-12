import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import requests
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import date, datetime, timedelta
import glob

# تثبيت القيم العشوائية
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

st.set_page_config(page_title="نموذج التنبؤ الذكي", layout="centered")
st.title("📊 هذا تطبيق تجريبي للتنبؤ — لا يمثل نصيحة مالية")

market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

if market == "🏦 السوق السعودي":
    ticker = user_input.upper() + ".SR"
elif market == "₿ العملات الرقمية":
    ticker = user_input.upper() + "-USD"
else:
    ticker = user_input.upper()

predict_days = st.selectbox("📆 عدد الأيام المستقبلية للتوقع:", [3, 5, 7])

forecast_path = f"forecasts/forecast_{ticker.replace('.', '_')}_{predict_days}.csv"
model_path = f"model_{ticker.replace('.', '_')}.h5"

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج..."):

        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)

        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات سعر الإغلاق (Close) لهذا الرمز.")
            st.stop()

        df.dropna(subset=['Close'], inplace=True)
        df['Close'] = df['Close'].astype(float)

        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()

        df.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'MACD']
        data = df[features].copy()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - predict_days):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i:i+predict_days, 3])  # Close

        X = np.array(X)
        y = np.array(y)

        # تحميل أو تدريب النموذج
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.info("✅ تم تحميل النموذج المحفوظ.")
        else:
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(predict_days))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=30, batch_size=32, verbose=0)
            model.save(model_path)
            st.success("✅ تم تدريب النموذج لأول مرة وتم حفظه.")

        # التحقق من وجود توقعات محفوظة
        os.makedirs("forecasts", exist_ok=True)
        if os.path.exists(forecast_path):
            forecast = pd.read_csv(forecast_path)['forecast'].values
            st.success("✅ تم تحميل التوقعات المحفوظة.")
        else:
            last_sequence = scaled_data[-sequence_length:]
            current_sequence = last_sequence.copy()
            forecast_scaled = []
            for _ in range(predict_days):
                prediction = model.predict(current_sequence.reshape(1, sequence_length, X.shape[2]), verbose=0)
                forecast_scaled.append(prediction[0][0])
                next_step = current_sequence[1:]
                next_close = prediction[0][0]
                next_row = current_sequence[-1].copy()
                next_row[3] = next_close  # Close
                current_sequence = np.vstack([next_step, next_row])

            inverse = scaler.inverse_transform(
                np.pad(np.zeros((predict_days, scaled_data.shape[1])), ((0,0), (0,0))).astype(float)
            )
            forecast_values = scaler.inverse_transform(
                np.hstack([np.zeros((predict_days, 3)),
                           np.array(forecast_scaled).reshape(-1,1),
                           np.zeros((predict_days, scaled_data.shape[1]-4))])
            )[:,3]
            pd.DataFrame({'forecast': forecast_values}).to_csv(forecast_path, index=False)
            forecast = forecast_values
            st.success("📁 تم حفظ التوقعات.")

        last_real = float(df['Close'].iloc[-1])
        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days+1, freq='B')[1:]

        st.subheader("📈 التوقعات القادمة:")
        for i, price in enumerate(forecast):
            color = 'green' if price > last_real else 'red'
            symbol = "↑" if price > last_real else "↓"
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:8px;color:white;'>اليوم {i+1}: {price:.2f} {symbol}</div>", unsafe_allow_html=True)

        st.subheader("📉 مقارنة السعر الحقيقي والتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'][-60:], label='السعر الحقيقي')
        ax.plot(forecast_dates, forecast, label='التوقع', linestyle='--', marker='o')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
