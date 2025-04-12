import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import datetime, timedelta, date
import random

# ✅ تثبيت العشوائية لثبات النتائج
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

st.set_page_config(page_title="نموذج التنبؤ الذكي", layout="centered")
st.title("📊 هذا تطبيق تجريبي للتنبؤ — لا يمثل نصيحة مالية")

symbol = st.text_input("🔍 أدخل رمز السهم أو العملة (مثال: AAPL أو BTC-USD أو 2222.SR)", "AAPL")
predict_days = st.selectbox("📆 عدد الأيام المستقبلية للتوقع:", [3, 5, 7])

model_path = f"models/model_{symbol.replace('.', '_')}_{predict_days}.h5"
forecast_path = f"forecasts/forecast_{symbol.replace('.', '_')}_{predict_days}.csv"

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)

        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لا توجد بيانات كافية أو عمود Close مفقود")
            st.stop()

        df.dropna(subset=['Close'], inplace=True)
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
        df.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'MACD']
        data = df[features]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled_data) - predict_days):
            X.append(scaled_data[i - seq_len:i])
            y.append(scaled_data[i:i + predict_days, 3])

        if len(X) == 0:
            st.error("⚠️ البيانات غير كافية لتدريب النموذج")
            st.stop()

        X, y = np.array(X), np.array(y)

        os.makedirs("models", exist_ok=True)
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.info("✅ تم تحميل النموذج المحفوظ")
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
            st.success("✅ تم تدريب النموذج وحفظه")

        os.makedirs("forecasts", exist_ok=True)
        if os.path.exists(forecast_path):
            forecast = pd.read_csv(forecast_path)['forecast'].values
            st.success("✅ تم تحميل التوقعات المحفوظة")
        else:
            last_seq = scaled_data[-seq_len:]
            curr_seq = last_seq.copy()
            preds = []

            for _ in range(predict_days):
                pred = model.predict(curr_seq.reshape(1, seq_len, X.shape[2]), verbose=0)
                preds.append(pred[0][0])
                next_row = curr_seq[-1].copy()
                next_row[3] = pred[0][0]
                curr_seq = np.vstack([curr_seq[1:], next_row])

            forecast_scaled = np.zeros((predict_days, scaled_data.shape[1]))
            forecast_scaled[:, 3] = preds
            forecast_prices = scaler.inverse_transform(forecast_scaled)[:, 3]
            pd.DataFrame({"forecast": forecast_prices}).to_csv(forecast_path, index=False)
            forecast = forecast_prices
            st.success("✅ تم حفظ التوقعات")

        last_price = df['Close'].iloc[-1]
        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days + 1, freq='B')[1:]

        st.subheader("📊 التوقعات:")
        for i, price in enumerate(forecast):
            color = 'green' if price > last_price else 'red'
            symbol = "↑" if price > last_price else "↓"
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:8px;color:white;'>اليوم {i + 1}: {price:.2f} {symbol}</div>", unsafe_allow_html=True)

        st.subheader("📉 مقارنة السعر الفعلي بالتوقع")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['Close'][-60:], label='السعر الحقيقي')
        ax.plot(forecast_dates, forecast, label='التوقع', linestyle='--', marker='o')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
