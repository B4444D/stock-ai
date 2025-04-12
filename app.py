# ✅ نسخة الكود التي تجعل التوقعات ثابتة ولا تتغير عند إعادة التشغيل

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
import glob

# تثبيت القيم العشوائية
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.set_page_config(page_title="نموذج التنبؤ بالأسعار", layout="centered")
st.title("📊 هذا تطبيق تجريبي للتنبؤ — لا يمثل نصيحة مالية")

symbol = st.text_input("🔍 أدخل رمز السهم أو العملة (مثال: AAPL أو BTC-USD أو 2222.SR)", "AAPL")
predict_days = st.selectbox("📆 عدد الأيام للتوقع:", [3, 5, 7])

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)

        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لا توجد بيانات كافية لهذا الرمز.")
            st.stop()

        df.dropna(subset=['Close'], inplace=True)
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
            y.append(scaled_data[i:i+predict_days, 3])  # العمود 3 = Close

        if len(X) == 0:
            st.warning("⚠️ لا توجد بيانات كافية لتدريب النموذج.")
            st.stop()

        X = np.array(X)
        y = np.array(y)

        # إذا كان النموذج محفوظ سابقًا استخدمه
        model_path = f"model_{symbol.replace('.', '_')}.h5"
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
            model.fit(X, y, epochs=30, batch_size=16, verbose=0)
            model.save(model_path)
            st.success("✅ تم تدريب النموذج وحفظه.")

        # التنبؤ
        forecast_path = f"forecasts/forecast_{symbol.replace('.', '_')}_{predict_days}.csv"
        os.makedirs("forecasts", exist_ok=True)

        if os.path.exists(forecast_path):
            forecast = pd.read_csv(forecast_path)['forecast'].values
            st.info("✅ تم تحميل التوقعات السابقة.")
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

            forecast = scaler.inverse_transform(
                np.pad(np.zeros((predict_days, scaled_data.shape[1])), ((0,0), (0,0)))
                .astype(float)
            )[:,3]
            forecast[:len(forecast_scaled)] = scaler.inverse_transform(
                np.hstack([np.zeros((predict_days, 3)),
                           np.array(forecast_scaled).reshape(-1,1),
                           np.zeros((predict_days, scaled_data.shape[1]-4))])
            )[:,3]

            pd.DataFrame({"forecast": forecast}).to_csv(forecast_path, index=False)
            st.success("✅ تم توليد التوقعات وحفظها.")

        last_price = df['Close'].iloc[-1]
        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days+1, freq='B')[1:]

        st.subheader("📈 التوقعات القادمة:")
        for i, price in enumerate(forecast):
            color = 'green' if price > last_price else 'red'
            symbol = "↑" if price > last_price else "↓"
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:8px;color:white;'>اليوم {i+1}: {price:.2f} {symbol}</div>", unsafe_allow_html=True)

        # رسم بياني
        st.subheader("📉 مقارنة السعر الفعلي والتوقع")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Close'], label='السعر الحقيقي')
        ax.plot(forecast_dates, forecast, label='التوقع', linestyle='--', marker='o')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
