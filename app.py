import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="تنبؤ الأسعار", layout="centered")
st.title("🔮 نموذج تنبؤ الأسعار مع السعر اللحظي")

api_key = "cvtcvi1r01qhup0vnjrgcvtcvi1r01qhup0vnjs0"

market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
symbol = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL").upper()
predict_days = st.selectbox("📅 عدد الأيام المستقبلية للتنبؤ:", [3, 5, 7])

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("📡 تحميل البيانات وتدريب النموذج..."):

        # الحصول على السعر اللحظي حسب السوق
        live_price = None
        if market == "🏦 السوق السعودي":
            ticker = symbol + ".SR"
            df = yf.download(ticker, period="6mo")
            try:
                live_price = float(df['Close'].dropna().iloc[-1])
            except:
                live_price = None
        elif market == "🇺🇸 السوق الأمريكي":
            ticker = symbol
            df = yf.download(ticker, period="6mo")
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
            r = requests.get(url).json()
            live_price = float(r["c"]) if "c" in r and r["c"] else None
        elif market == "₿ العملات الرقمية":
            ticker = symbol + "-USD"
            df = yf.download(ticker, period="6mo")
            url = f"https://finnhub.io/api/v1/quote?symbol=BINANCE:{symbol}USDT&token={api_key}"
            r = requests.get(url).json()
            live_price = float(r["c"]) if "c" in r and r["c"] else None

        if df.empty:
            st.error("❌ لا توجد بيانات كافية.")
            st.stop()

        # إضافة مؤشرات RSI و MACD
        import ta
        close_clean = pd.Series(df['Close'].values, index=df.index).astype(float)
df['RSI'] = ta.momentum.RSIIndicator(close=close_clean, window=14).rsi().fillna(0)
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd().fillna(0)

        df = df[['Close', 'RSI', 'MACD']].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)

        sequence_len = 60
        X, y = [], []
        for i in range(sequence_len, len(scaled) - predict_days):
            X.append(scaled[i-sequence_len:i])
            y.append(scaled[i:i+predict_days, 0])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(sequence_len, input_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(predict_days))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        last_seq = scaled[-sequence_len:]
        input_features = scaled.shape[1]
        preds_scaled = model.predict(last_seq.reshape(1, sequence_len, 1))[0]
        forecast = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        real_price = live_price if live_price else df['Close'].iloc[-1]

        st.subheader("🔮 التوقعات:")
        for i, price in enumerate(forecast):
            color = 'green' if price > real_price else 'red'
            arrow = "📈" if price > real_price else "📉"
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;'>اليوم {i+1}: {price:.2f} {arrow}</div>", unsafe_allow_html=True)

        st.subheader("📊 السعر الحقيقي المستخدم للمقارنة:")
        st.info(f"{real_price:.2f}")
