import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import date
import glob

# ♥️ تثبيت القيم العشوائية لجعل التوقعات ثابتة
np.random.seed(42)

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

predict_days = st.selectbox("🗖️ عدد الأيام المستقبلية للتوقع:", [3, 5, 7])

def get_crypto_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd&include_24hr_change=true"
    response = requests.get(url)
    try:
        data = response.json()
        price = data[symbol]['usd']
        change = data[symbol]['usd_24h_change']
        return float(price), float(change)
    except:
        return None, None

if st.button("🚀 ابدأ التنبؤ"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج..."):

        if market == "₿ العملات الرقمية":
            live_price, _ = get_crypto_price(user_input.lower())
        else:
            live_price = None

        df = yf.download(ticker, start="2021-01-01")

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات سعر الإغلاق (Close) لهذا الرمز.")
            st.write("📋 الأعمدة المتوفرة في البيانات:", df.columns.tolist())
            st.stop()

        df = df[df['Close'].notna()]
        df['Close'] = df['Close'].astype(float)

        clean_close = df['Close'].copy()
        clean_close = pd.Series(clean_close.values.flatten(), index=df.index).astype(float)

        df['RSI'] = ta.momentum.RSIIndicator(close=clean_close).rsi()
        df['EMA20'] = ta.trend.EMAIndicator(close=clean_close, window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(close=clean_close, window=50).ema_indicator()
        macd = ta.trend.MACD(close=clean_close)
        df['MACD'] = macd.macd()

        try:
            stoch = ta.momentum.StochasticOscillator(
                high=df['High'],
                low=df['Low'],
                close=clean_close
            )
            df['Stoch_K'] = stoch.stoch().fillna(0)
            df['Stoch_D'] = stoch.stoch_signal().fillna(0)
        except Exception as e:
            st.warning(f"⚠️ تعذر حساب مؤشر Stochastic: {e}")
            df['Stoch_K'] = 0
            df['Stoch_D'] = 0

        df.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'MACD', 'Stoch_K', 'Stoch_D']
        data = df[features]
        scalers = {}
        scaled_data = pd.DataFrame(index=data.index)
        for col in features:
            if col not in data.columns or data[col].dropna().shape[0] == 0:
                st.warning(f"⚠️ تم تجاهل العمود '{col}' لأنه لا يحتوي على بيانات قابلة للاستخدام.")
                continue
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

        if scaled_data.shape[1] == 0:
            st.error("❌ لا توجد بيانات كافية للتدريب. حاول تغيير الرمز أو توسيع الفترة الزمنية.")
            st.stop()

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)-predict_days):
            X.append(scaled_data.iloc[i-sequence_length:i].values)
            y.append(scaled_data.iloc[i:i+predict_days]['Close'].values)

        if len(X) == 0:
            st.error("⚠️ البيانات غير كافية لتدريب النموذج. يرجى تجربة رمز آخر أو فترة زمنية أطول.")
            st.stop()

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(predict_days))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)

        last_sequence = scaled_data[-sequence_length:].values
        forecast_scaled = []
        current_sequence = last_sequence.copy()

        for _ in range(predict_days):
            prediction = model.predict(current_sequence.reshape(1, sequence_length, scaled_data.shape[1]), verbose=0)
            forecast_scaled.append(prediction[0][0])
            next_step = current_sequence[1:]
            next_row = current_sequence[-1].copy()
            close_idx = scaled_data.columns.get_loc('Close')
            next_row[close_idx] = prediction[0][0]
            current_sequence = np.vstack([next_step, next_row])

        forecast = scalers['Close'].inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        last_real = float(df['Close'].iloc[-1])

        if live_price:
            st.info(f"💲 السعر اللحظي من الإنترنت: {live_price:.2f}")
        else:
            st.info(f"🔒 السعر الأخير للإغلاق: {last_real:.2f}")

        st.subheader("📈 التوقعات القادمة:")
        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days+1, freq='B')[1:]
        for i, price in enumerate(forecast):
            color = 'green' if price > last_real else 'red'
            symbol = "↑" if price > last_real else "↓"
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:8px;color:white;'>اليوم {i+1}: {price:.2f} {symbol}</div>", unsafe_allow_html=True)

        os.makedirs("forecasts", exist_ok=True)
        result_df = pd.DataFrame({'date': forecast_dates, 'predicted_close': forecast})
        save_path = f"forecasts/forecast_{ticker.replace('.', '_')}_{date.today()}.csv"
        result_df.to_csv(save_path, index=False)
        st.success(f"✅ تم حفظ التوقعات في ملف: {save_path}")

        st.subheader("📉 مقارنة السعر الحقيقي والتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'][-100:], label='السعر الحقيقي')
        ax.plot(forecast_dates, forecast, label='التوقع', linestyle='--', marker='o')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
