import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="سعر السوق اللحظي", layout="centered")
st.title("📊 جلب السعر اللحظي من الأسواق")

# إعداد مفتاح Alpha Vantage
api_key = "EPOU1W12WSZL18ST"

# اختيار السوق والرمز
market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

# زر تنفيذ
if st.button("📥 جلب السعر اللحظي"):
    if market == "🏦 السوق السعودي":
        ticker = user_input.upper() + ".SR"
        try:
            df = yf.download(ticker, period="1d", interval="1m")
            if not df.empty and 'Close' in df.columns:
                last_price = df['Close'].dropna().iloc[-1]
                st.success(f"✅ السعر اللحظي لـ {user_input.upper()}: {last_price:.2f} ريال")
            else:
                st.warning("⚠️ لم يتم العثور على بيانات حديثة.")
        except:
            st.error("❌ تعذر تحميل السعر من yfinance")

    elif market == "🇺🇸 السوق الأمريكي":
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={user_input.upper()}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            try:
                price = float(data["Global Quote"]["05. price"])
                st.success(f"✅ السعر اللحظي لـ {user_input.upper()}: {price:.2f} دولار")
            except:
                st.warning("⚠️ فشل في تحويل السعر.")
        else:
            st.warning("⚠️ تعذر العثور على السعر. تأكد من الرمز.")

    elif market == "₿ العملات الرقمية":
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={user_input.upper()}&market=USD&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "Time Series (Digital Currency Daily)" in data:
            try:
                latest = list(data["Time Series (Digital Currency Daily)"].values())[0]
                price = float(latest['1a. open (USD)'])
                st.success(f"✅ السعر اللحظي لـ {user_input.upper()}: {price:.2f} دولار")
            except:
                st.warning("⚠️ فشل في قراءة بيانات العملة.")
        else:
            st.warning("⚠️ تعذر جلب بيانات العملة الرقمية. تأكد من الرمز مثل BTC أو ETH.")
