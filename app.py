import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="سعر السوق اللحظي", layout="centered")
st.title("📊 جلب السعر اللحظي من الأسواق")

# مفتاح Alpha Vantage
api_key = "EPOU1W12WSZL18ST"

# تحديد السوق والرمز
market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

# زر تنفيذ العملية
if st.button("📥 جلب السعر اللحظي"):
    symbol = user_input.upper()

    # 🏦 السوق السعودي
    if market == "🏦 السوق السعودي":
        ticker = symbol + ".SR"
        try:
            df = yf.download(ticker, period="5d", interval="15m")
            if not df.empty and 'Close' in df.columns:
                last_price = df['Close'].dropna().iloc[-1]
                st.success(f"✅ السعر اللحظي لـ {symbol}: {last_price:.2f} ريال")
            else:
                st.warning("⚠️ لم يتم العثور على بيانات حديثة.")
        except Exception as e:
            st.error(f"❌ خطأ أثناء تحميل البيانات: {e}")

    # 🇺🇸 السوق الأمريكي
    elif market == "🇺🇸 السوق الأمريكي":
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            try:
                price = float(data["Global Quote"]["05. price"])
                st.success(f"✅ السعر اللحظي لـ {symbol}: {price:.2f} دولار")
            except:
                st.warning("⚠️ فشل في تحويل السعر.")
        else:
            st.warning("⚠️ تعذر العثور على السعر. تأكد من الرمز.")

    # ₿ العملات الرقمية
    elif market == "₿ العملات الرقمية":
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "Time Series (Digital Currency Daily)" in data:
            try:
                latest_day = list(data["Time Series (Digital Currency Daily)"].keys())[0]
                latest_price = data["Time Series (Digital Currency Daily)"][latest_day]["4a. close (USD)"]
                price = float(latest_price)
                st.success(f"✅ السعر الحالي لـ {symbol}: {price:.2f} دولار")
            except:
                st.warning("⚠️ فشل في قراءة بيانات العملة الرقمية.")
        else:
            st.warning("⚠️ تعذر جلب بيانات العملة الرقمية. تأكد من الرمز مثل BTC أو ETH.")
