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

        df = df[df['Close'].notna()]
        df['Close'] = df['Close'].astype(float)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].fillna(df[col].rolling(5, min_periods=1).mean())

        clean_close = df['Close'].copy()
        if isinstance(clean_close, pd.DataFrame):
            clean_close = clean_close.iloc[:, 0]
        clean_close = pd.Series(clean_close.values, index=df.index).astype(float)

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
            high = df['High'].values.flatten()
            low = df['Low'].values.flatten()
            close = clean_close.values.flatten()
            stoch = ta.momentum.StochasticOscillator(
                high=pd.Series(high, index=df.index),
                low=pd.Series(low, index=df.index),
                close=pd.Series(close, index=df.index),
                window=14,
                smooth_window=3
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        except Exception as e:
            st.error(f"❌ حدث خطأ في حساب المؤشرات الفنية: {str(e)}")
            st.stop()

        df = df.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 
                   'EMA20', 'EMA50', 'EMA200', 'BB_upper', 'BB_middle', 
                   'BB_lower', 'MACD', 'MACD_signal', 'MACD_diff', 
                   'Stoch_K', 'Stoch_D', 'VWAP']

        scalers = {}
        scaled_data = pd.DataFrame(index=df.index)
        for col in features:
            try:
                if df[col].dropna().shape[0] == 0:
                    st.warning(f"⚠️ تم تجاهل العمود '{col}' لأنه لا يحتوي على بيانات قابلة للاستخدام.")
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data[col] = scaler.fit_transform(df[[col]]).flatten()
                scalers[col] = scaler
            except Exception as e:
                st.warning(f"⚠️ تعذر تحجيم العمود '{col}': {str(e)} — تم تجاهله.")
                continue

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)-predict_days):
            X.append(scaled_data.iloc[i-sequence_length:i].values)
            y.append(scaled_data.iloc[i:i+predict_days]['Close'].values)

        X, y = np.array(X), np.array(y)
        if len(X) == 0:
            st.error("❌ لا توجد بيانات كافية للتدريب. حاول زيادة عدد الأيام التاريخية.")
            st.stop()

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
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        st.success(f"✅ تم تدريب النموذج بنجاح | فقدان التدريب: {train_loss:.4f} | فقدان التحقق: {val_loss:.4f}")

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

        st.subheader("📊 نتائج التنبؤ المحسنة")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("آخر سعر إغلاق", f"{last_real:.2f}")
        with col2:
            change_percent = (forecast[-1] - last_real) / last_real * 100
            st.metric("التوقع النهائي", 
                     f"{forecast[-1]:.2f}",
                     delta=f"{change_percent:.2f}%",
                     delta_color="normal")

        st.subheader("📅 التوقعات اليومية")
        forecast_dates = pd.date_range(start=df.index[-1], periods=predict_days+1, freq='B')[1:]
        for i, (date, price) in enumerate(zip(forecast_dates, forecast)):
            change = (price - last_real) / last_real * 100
            arrow = "↑" if change >= 0 else "↓"
            color = "green" if change >= 0 else "red"
            st.markdown(f"""
            <div style='border-left: 5px solid {color}; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                <b>{date.strftime('%Y-%m-%d')}</b> (يوم {i+1}): 
                <span class='{"positive" if change >=0 else "negative"}'>
                    {price:.2f} {arrow} ({change:.2f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📉 مقارنة تاريخية مع التوقعات")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'][-30:], label='السعر التاريخي', color='blue', linewidth=2)
        ax.plot(forecast_dates, forecast, label='التوقعات', linestyle='--', marker='o', color='green', linewidth=2)
        ax.fill_between(forecast_dates, forecast * 0.97, forecast * 1.03, color='green', alpha=0.1, label='منطقة الثقة ±3%')
        ax.axhline(last_real, color='red', linestyle=':', label='آخر سعر معروف')
        ax.set_title(f"توقعات أسعار {ticker} للـ {predict_days} أيام القادمة")
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر (USD)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        sns.despine()
        st.pyplot(fig)

        os.makedirs("forecasts", exist_ok=True)
        result_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_close': forecast,
            'confidence_lower': forecast * 0.97,
            'confidence_upper': forecast * 1.03
        })
        save_path = f"forecasts/forecast_{ticker.replace('.', '_')}_{date.today()}.csv"
        result_df.to_csv(save_path, index=False)
        st.success(f"💾 تم حفظ التوقعات في: {save_path}")

        st.subheader("🔍 تحليل دقة التوقعات السابقة")
        review_files = glob.glob("forecasts/forecast_*.csv")
        review_results = []

        for file in sorted(review_files, reverse=True)[:5]:
            try:
                df_forecast = pd.read_csv(file)
                df_forecast['date'] = pd.to_datetime(df_forecast['date'])
                real_data = yf.download(ticker, 
                                      start=df_forecast['date'].min() - timedelta(days=3),
                                      end=df_forecast['date'].max() + timedelta(days=3))
                if real_data.empty:
                    continue
                temp_results = []
                for _, row in df_forecast.iterrows():
                    real_price = real_data.loc[row['date'].strftime('%Y-%m-%d')]['Close']
                    error = abs(real_price - row['predicted_close'])
                    accuracy = max(0, 100 - (error / real_price * 100))
                    temp_results.append({
                        'الرمز': ticker,
                        'التاريخ': row['date'].strftime('%Y-%m-%d'),
                        'التوقع': round(row['predicted_close'], 2),
                        'الحقيقي': round(real_price, 2),
                        'الخطأ': round(error, 2),
                        'الدقة (%)': round(accuracy, 2),
                        'ملاحظات': '✅ ضمن الثقة' if row['confidence_lower'] <= real_price <= row['confidence_upper'] else '❌ خارج الثقة'
                    })
                review_results.extend(temp_results)
            except Exception as e:
                continue

        if review_results:
            review_df = pd.DataFrame(review_results)
            st.dataframe(review_df.sort_values('التاريخ', ascending=False))
            avg_accuracy = review_df['الدقة (%)'].mean()
            st.metric("متوسط دقة التوقعات السابقة", f"{avg_accuracy:.2f}%")
        else:
            st.info("لا توجد بيانات كافية لتحليل الدقة السابقة. ستصبح هذه الميزة أكثر فائدة بعد عدة تنبؤات.")
