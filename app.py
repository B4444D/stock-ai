import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="عرض بيانات السهم", layout="centered")
st.title("📊 تطبيق عرض بيانات السهم")

# اختيار السوق
market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

# تجهيز الرمز
if market == "🏦 السوق السعودي":
    ticker = user_input.upper() + ".SR"
elif market == "₿ العملات الرقمية":
    ticker = user_input.upper() + "-USD"
else:
    ticker = user_input.upper()

# زر البدء
if st.button("📥 تحميل البيانات"):
    with st.spinner("جاري تحميل البيانات..."):
        df = yf.download(ticker, start="2021-01-01")

        if df.empty:
            st.error("❌ لم يتم العثور على بيانات.")
        else:
            st.success("✅ تم تحميل البيانات بنجاح!")

            st.subheader("🔚 السعر الأخير:")
            valid_closes = df['Close'].dropna()
            if not valid_closes.empty:
                last_close = valid_closes.iloc[-1]
                st.write(f"آخر إغلاق: {last_close:.2f}")
            else:
                st.warning("⚠️ لا توجد بيانات إغلاق متوفرة.")

            st.subheader("📈 الرسم البياني:")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['Close'], label='سعر الإغلاق')
            ax.set_title(f"أداء {ticker}")
            ax.grid()
            st.pyplot(fig)
