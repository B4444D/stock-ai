import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import date, timedelta
import glob

# إعداد واجهة المستخدم
st.set_page_config(page_title="نموذج التنبؤ الذكي المحسن", layout="centered")
st.title("📊 نظام التنبؤ المالي الذكي - النسخة المحسنة")
st.markdown("""
<style>
.positive { color: green; font-weight: bold; }
.negative { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# إعداد خيارات السوق
market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

if market == "🏦 السوق السعودي":
    ticker = user_input.upper() + ".SR"
elif market == "₿ العملات الرقمية":
    ticker = user_input.upper() + "-USD"
else:
    ticker = user_input.upper()

predict_days = st.selectbox("📆 عدد الأيام المستقبلية للتوقع:", [3, 5, 7, 10])
lookback_days = st.slider("↩️ عدد الأيام للرجوع للبيانات التاريخية:", 60, 365, 180)

# دالة محسنة لجلب بيانات العملات الرقمية
def get_crypto_price(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart?vs_currency=usd&days=30"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        latest_price = df['price'].iloc[-1]
        change_24h = (latest_price - df['price'].iloc[-2]) / df['price'].iloc[-2] * 100
        return float(latest_price), float(change_24h)
    except Exception as e:
        st.warning(f"⚠️ خطأ في جلب بيانات العملة الرقمية: {str(e)}")
        return None, None

if st.button("🚀 ابدأ التنبؤ المحسن"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج المتقدم..."):
        start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        if market == "₿ العملات الرقمية":
            live_price, price_change = get_crypto_price(user_input.lower())
            if live_price and price_change:
                st.info(f"💲 السعر اللحظي: {live_price:.2f} USD | التغير 24h: {'+' if price_change > 0 else ''}{price_change:.2f}%")
            df = yf.download(ticker, start=start_date, progress=False)
        else:
            df = yf.download(ticker, start=start_date, progress=False)
            live_price = None

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات لهذا الرمز أو هناك مشكلة في الاتصال.")
            st.stop()

        df = df[df['Close'].notna()]
        df['Close'] = df['Close'].astype(float)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].fillna(df[col].rolling(5, min_periods=1).mean())

        clean_close = df['Close'].copy()
        clean_close = pd.Series(clean_close.values.flatten(), index=df.index).astype(float)

        df['RSI'] = ta.momentum.RSIIndicator(close=clean_close, window=14).rsi()
        df['EMA20'] = ta.trend.EMAIndicator(close=clean_close, window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(close=clean_close, window=50).ema_indicator()
        df['EMA200'] = ta.trend.EMAIndicator(close=clean_close, window=200).ema_indicator()

        bb = ta.volatility.BollingerBands(close=clean_close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()

        macd = ta.trend.MACD(close=clean_close, window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        try:
            df['Stoch_K'] = ta.momentum.StochasticOscillator(
                high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
            ).stoch().fillna(50).astype(float).values.flatten()
            df['Stoch_D'] = ta.momentum.StochasticOscillator(
                high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
            ).stoch_signal().fillna(50).astype(float).values.flatten()
        except Exception as e:
            st.warning(f"⚠️ تعذر حساب مؤشر Stochastic: {e}")
            df['Stoch_K'] = 50
            df['Stoch_D'] = 50

        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        df.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 
                    'EMA20', 'EMA50', 'EMA200', 'BB_upper', 'BB_middle', 
                    'BB_lower', 'MACD', 'MACD_signal', 'MACD_diff', 
                    'Stoch_K', 'Stoch_D', 'VWAP']

        scalers = {}
        scaled_data = pd.DataFrame(index=df.index)
        for col in features:
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_col = scaler.fit_transform(df[[col]])
                scaled_data[col] = scaled_col.flatten()
                scalers[col] = scaler
            except Exception as e:
                st.warning(f"⚠️ تعذر تحجيم العمود {col}: {e}")
                continue

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)-predict_days):
            X.append(scaled_data.iloc[i-sequence_length:i].values)
            y.append(scaled_data.iloc[i:i+predict_days]['Close'].values)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = Sequential([
            LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.3),
            LSTM(150, return_sequences=False),
            Dropout(0.3),
            Dense(predict_days)
        ])

        model.compile(optimizer='adam', loss='huber_loss', metrics=['mae'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        st.success(f"✅ تم تدريب النموذج بنجاح | فقدان التدريب: {train_loss:.4f} | فقدان التحقق: {val_loss:.4f}")
