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

        # تحميل بيانات آخر شهرين فقط
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات سعر الإغلاق (Close) لهذا الرمز.")
            st.write("📋 الأعمدة المتوفرة في البيانات:", df.columns.tolist())
            st.stop()

        df = df[df['Close'].notna()]
        df['Close'] = df['Close'].astype(float)

        clean_close = df['Close'].copy()
        if isinstance(clean_close, pd.DataFrame):
            clean_close = clean_close.iloc[:, 0]
        clean_close = pd.Series(clean_close.values, index=df.index).astype(float)

        df['RSI'] = ta.momentum.RSIIndicator(close=clean_close).rsi()
        df['EMA20'] = ta.trend.EMAIndicator(close=clean_close, window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(close=clean_close, window=50).ema_indicator()
        macd = ta.trend.MACD(close=clean_close)
        df['MACD'] = macd.macd()

        try:
            high = np.squeeze(df['High'].values)
            low = np.squeeze(df['Low'].values)
            close = np.squeeze(clean_close.values)

            stoch = ta.momentum.StochasticOscillator(
                high=pd.Series(high, index=df.index),
                low=pd.Series(low, index=df.index),
                close=pd.Series(close, index=df.index)
            )
            stoch_k = stoch.stoch().fillna(0)
            stoch_d = stoch.stoch_signal().fillna(0)
            df['Stoch_K'] = stoch_k.values
            df['Stoch_D'] = stoch_d.values
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
            if data[col].isnull().any() or data[col].dropna().shape[0] == 0:
                st.warning(f"⚠️ العمود '{col}' يحتوي على بيانات غير كافية أو قيم مفقودة وتم تجاهله.")
                continue
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

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

        if os.path.exists("trained_model.h5"):
            model = load_model("trained_model.h5")
            st.info("✅ تم تحميل النموذج المدرب مسبقًا.")
        else:
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(predict_days))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
            model.save("trained_model.h5")
            st.success("✅ تم تدريب النموذج لأول مرة وحفظه.")

        last_sequence = scaled_data[-sequence_length:].values
        forecast_scaled = []
        current_sequence = last_sequence.copy()

        for _ in range(predict_days):
            prediction = model.predict(current_sequence.reshape(1, sequence_length, len(features)), verbose=0)
            forecast_scaled.append(prediction[0][0])
            next_step = current_sequence[1:]
            next_close = prediction[0][0]
            next_row = current_sequence[-1].copy()
            next_row[features.index('Close')] = next_close
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
