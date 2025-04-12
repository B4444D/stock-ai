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
            if live_price:
                st.info(f"💲 السعر اللحظي: {live_price:.2f} USD | التغير 24h: {'+' if price_change > 0 else ''}{price_change:.2f}%")
            df = yf.download(ticker, start=start_date, progress=False)
        else:
            df = yf.download(ticker, start=start_date, progress=False)
            live_price = None

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات لهذا الرمز أو هناك مشكلة في الاتصال.")
            st.stop()

        if df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().shape[0] < 60:
            st.error("⚠️ البيانات غير كافية بعد المعالجة. جرب رمزًا آخر أو زِد عدد الأيام.")
            st.stop()

        # الباقي من الكود يبقى كما هو...
