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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import date
import glob

# تثبيت العشوائية لضمان ثبات النتائج
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

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

def get_crypto_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd&include_24hr_change=true"
    response = requests.get(url)
    try:
        data = response.json()
        price = data[symbol]['usd']
        change = data[symbol]['usd_24hr_change']
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

# المؤشرات الفنية (خارج أي تداخل)
clean_close_for_rsi = df['Close'].reset_index(drop=True).astype(float).fillna(method='ffill')
rsi_indicator = ta.momentum.RSIIndicator(close=clean_close_for_rsi, window=14)
df['RSI'] = rsi_indicator.rsi().reindex(df.index).fillna(0)

df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator().fillna(0)
df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator().fillna(0)

macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd().fillna(0)

try:
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close']
    )
    df['Stoch_K'] = stoch.stoch().fillna(0)
    df['Stoch_D'] = stoch.stoch_signal().fillna(0)
except Exception as e:
    st.warning(f"⚠️ تعذر حساب مؤشر Stochastic: {e}")
    df['Stoch_K'] = 0
    df['Stoch_D'] = 0

        clean_close = df['Close'].dropna().astype(float).fillna(method='ffill')

        # المؤشرات الفنية بتنسيق سليم
            clean_close_for_rsi = df['Close'].reset_index(drop=True).astype(float).fillna(method='ffill')
rsi_indicator = ta.momentum.RSIIndicator(close=clean_close_for_rsi, window=14)
df['RSI'] = rsi_indicator.rsi().reindex(df.index).fillna(0)

df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator().fillna(0)
df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator().fillna(0)

macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd().fillna(0)

try:
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close']
    )
    df['Stoch_K'] = stoch.stoch().fillna(0)
    df['Stoch_D'] = stoch.stoch_signal().fillna(0)
except Exception as e:
    st.warning(f"⚠️ تعذر حساب مؤشر Stochastic: {e}")
    df['Stoch_K'] = 0
    df['Stoch_D'] = 0
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
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)-predict_days):
            X.append(scaled_data.iloc[i-sequence_length:i].values)
            y.append(scaled_data.iloc[i:i+predict_days]['Close'].values)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dropout(0.3))
        model.add(Dense(predict_days))
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)

        last_sequence = scaled_data[-sequence_length:].values
        forecast_scaled = []
        current_sequence = last_sequence.copy()

        for _ in range(predict_days):
            prediction = model.predict(current_sequence.reshape(1, sequence_length, len(features)), verbose=0)
            forecast_scaled.append(prediction[0][0])
            next_step = current_sequence[1:]
            next_row = current_sequence[-1].copy()
            next_row[features.index('Close')] = prediction[0][0]
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

        st.subheader("📋 مراجعة التوقعات السابقة")
        review_files = glob.glob("forecasts/forecast_*.csv")
        review_results = []

        for file in review_files:
            try:
                df_forecast = pd.read_csv(file)
                forecast_dates = pd.to_datetime(df_forecast['date'])
                predicted = df_forecast['predicted_close']

                real_data = yf.download(ticker, start=str(forecast_dates.min().date()), end=str(forecast_dates.max().date()))
                if real_data.empty:
                    continue
                real_prices = real_data['Close']

                for i, f_date in enumerate(forecast_dates):
                    real_price = real_prices.get(f_date.strftime("%Y-%m-%d"), None)
                    if real_price:
                        predicted_price = predicted[i]
                        error = abs(real_price - predicted_price)
                        accuracy = 100 - (error / real_price * 100)
                        review_results.append({
                            'التاريخ': f_date.date(),
                            'السعر المتوقع': round(predicted_price, 2),
                            'السعر الحقيقي': round(real_price, 2),
                            'الدقة (%)': round(accuracy, 2)
                        })
            except:
                continue

        if review_results:
            st.dataframe(pd.DataFrame(review_results).sort_values("التاريخ", ascending=False))
        else:
            st.info("📭 لا توجد توقعات سابقة لمراجعتها حتى الآن.")
