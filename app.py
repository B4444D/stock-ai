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
