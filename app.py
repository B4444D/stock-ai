import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import date, timedelta
import glob

# تثبيت القيم العشوائية
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# إعداد واجهة المستخدم
st.set_page_config(page_title="نموذج التنبؤ الذكي التجريبي", layout="centered")
st.title("📊 نظام التنبؤ المالي الذكي - النسخة التجريبة")
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

# جلب البيانات وتحليلها عند الضغط
if st.button("🚀 ابدأ التنبؤ المحسن"):
    with st.spinner("جاري تحميل البيانات وتدريب النموذج المتقدم..."):
        start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_date, progress=False)

        if df.empty or 'Close' not in df.columns:
            st.error("❌ لم يتم العثور على بيانات لهذا الرمز أو هناك مشكلة في الاتصال.")
            st.stop()

        df = df[df['Close'].notna()].copy()
        df['Close'] = df['Close'].astype(float)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].fillna(df[col].rolling(5, min_periods=1).mean())

        clean_close = df['Close']

        # المؤشرات الفنية
        try:
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
            stoch = ta.momentum.StochasticOscillator(
                high=df['High'], low=df['Low'], close=clean_close,
                window=14, smooth_window=3
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        except Exception as e:
            st.error(f"❌ حدث خطأ في حساب المؤشرات الفنية: {str(e)}")
            st.stop()

        df.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'EMA200',
                    'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_diff',
                    'Stoch_K', 'Stoch_D', 'VWAP']

        scalers, scaled_data = {}, pd.DataFrame(index=df.index)
        for col in features:
            try:
                if col not in df.columns or df[col].dropna().shape[0] == 0:
                    st.warning(f"⚠️ تم تجاهل العمود '{col}' لأنه لا يحتوي على بيانات قابلة للاستخدام.")
                    continue
                scaler = MinMaxScaler()
                scaled_data[col] = scaler.fit_transform(df[[col]]).flatten()
                scalers[col] = scaler
            except Exception as e:
                st.warning(f"⚠️ تعذر تحجيم العمود {col}: {str(e)}")

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - predict_days):
            X.append(scaled_data.iloc[i - sequence_length:i].values)
            y.append(scaled_data.iloc[i:i + predict_days]['Close'].values)

        X, y = np.array(X), np.array(y)
        if len(X) == 0:
            st.error("❌ لا توجد بيانات كافية للتدريب. حاول زيادة عدد الأيام التاريخية.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model_file = f"models/model_{ticker.replace('.', '_')}_{predict_days}.h5"
        if os.path.exists(model_file):
            model = load_model(model_file)
            st.info("✅ تم تحميل النموذج المدرب مسبقًا.")
        else:
            model = Sequential([
                LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.3),
                LSTM(150),
                Dropout(0.3),
                Dense(predict_days)
            ])
            model.compile(optimizer='adam', loss='huber_loss')
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            os.makedirs("models", exist_ok=True)
            model.save(model_file)

        last_sequence = scaled_data[-sequence_length:].values
        forecast_scaled = []
        current_sequence = last_sequence.copy()

        for _ in range(predict_days):
            prediction = model.predict(current_sequence.reshape(1, sequence_length, len(scaled_data.columns)), verbose=0)
            forecast_scaled.append(prediction[0][0])
            next_step = current_sequence[1:]
            next_row = current_sequence[-1].copy()
            next_row[scaled_data.columns.get_loc('Close')] = prediction[0][0]
            current_sequence = np.vstack([next_step, next_row])

        forecast = scalers['Close'].inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        last_real = float(df['Close'].iloc[-1])

        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days + 1, freq='B')[1:]
        st.subheader("📅 التوقعات اليومية")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast)):
            change = (price - last_real) / last_real * 100
            arrow = "↑" if change >= 0 else "↓"
            color = "green" if change >= 0 else "red"
            st.markdown(f"""
            <div style='border-left: 5px solid {color}; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                <b>{date.strftime('%Y-%m-%d')}</b> (يوم {i+1}): 
                <span class='{'positive' if change >=0 else 'negative'}'>
                    {price:.2f} {arrow} ({change:.2f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'][-30:], label='السعر التاريخي', color='blue')
        ax.plot(forecast_dates, forecast, '--o', color='green', label='التوقعات')
        ax.axhline(last_real, color='red', linestyle=':', label='آخر سعر')
        ax.fill_between(forecast_dates, forecast * 0.97, forecast * 1.03, color='green', alpha=0.1)
        ax.legend()
        st.pyplot(fig)

        result_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_close': forecast,
            'confidence_lower': forecast * 0.97,
            'confidence_upper': forecast * 1.03
        })
        os.makedirs("forecasts", exist_ok=True)
        result_df.to_csv(f"forecasts/forecast_{ticker.replace('.', '_')}_{date.today()}.csv", index=False)
        st.success("✅ تم حفظ التوقعات بنجاح.")
