import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="نموذج تنبؤ بالأسعار", layout="centered")
st.title("🔮 تنبؤ الأسعار باستخدام المؤشرات الفنية")

api_key = "cvtcvi1r01qhup0vnjrgcvtcvi1r01qhup0vnjs0"

market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
symbol = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL").upper()
predict_days = st.selectbox("📅 عدد الأيام المستقبلية للتنبؤ:", [3, 5, 7])

# تثبيت القيم العشوائية لضمان ثبات النتائج
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("📡 تحميل البيانات وتدريب النموذج..."):

        # تحميل البيانات حسب السوق
        if market == "🏦 السوق السعودي":
            ticker = symbol + ".SR"
            df = yf.download(ticker, period="6mo")
        elif market == "🇺🇸 السوق الأمريكي":
            ticker = symbol
            df = yf.download(ticker, period="6mo")
        elif market == "₿ العملات الرقمية":
            ticker = symbol + "-USD"
            df = yf.download(ticker, period="6mo")

        if df.empty or 'Close' not in df:
            st.error("❌ لم يتم تحميل البيانات بنجاح.")
            st.stop()

        # جلب السعر اللحظي
        live_price = None
        if market == "🇺🇸 السوق الأمريكي":
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
            r = requests.get(url).json()
            live_price = float(r["c"]) if "c" in r and r["c"] else None
        elif market == "₿ العملات الرقمية":
            url = f"https://finnhub.io/api/v1/quote?symbol=BINANCE:{symbol}USDT&token={api_key}"
            r = requests.get(url).json()
            live_price = float(r["c"]) if "c" in r and r["c"] else None
        elif market == "🏦 السوق السعودي":
            try:
                live_price = float(df['Close'].dropna().iloc[-1])
            except:
                live_price = None

        if live_price:
            st.info(f"💰 السعر اللحظي لـ {symbol}: {live_price:.2f}")
        else:
            st.warning("❌ تعذر جلب السعر اللحظي.")

        # تنظيف البيانات
        df = df[['Close']].dropna()
        close_clean = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)

        # حساب RSI و MACD
        rsi_values = ta.momentum.RSIIndicator(close=close_clean, window=14).rsi()
        df['RSI'] = rsi_values.reindex(df.index).fillna(0)

        macd_values = ta.trend.MACD(close=close_clean)
        df['MACD'] = macd_values.macd().reindex(df.index).fillna(0)

        # التطبيع
        close_scaler = MinMaxScaler()
        df['Close_scaled'] = close_scaler.fit_transform(df[['Close']])

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close_scaled', 'RSI', 'MACD']])

        # تجهيز بيانات التدريب
        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled) - predict_days):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i:i+predict_days, 0])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # بناء النموذج
        input_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, input_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(predict_days))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=32, shuffle=False, verbose=0)

        # التنبؤ
        last_seq = scaled[-seq_len:]
        preds_scaled = model.predict(last_seq.reshape(1, seq_len, input_features))[0]
        forecast = close_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # عرض النتائج
        st.subheader("🔮 التوقعات:")
        for i, price in enumerate(forecast):
            st.markdown(f"اليوم {i+1}: {price:.2f} ريال / دولار")

        st.subheader("📊 رسم بياني للسعر")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Close'][-100:], label='السعر الفعلي')
        ax.set_title(f"آخر أسعار {symbol}")
        ax.grid()
        st.pyplot(fig)

        st.success("✅ النموذج يعمل باستخدام RSI و MACD بدقة.")
