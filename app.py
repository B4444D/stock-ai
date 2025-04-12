import streamlit as st
import requests
import yfinance as yf

st.set_page_config(page_title="📈 السعر اللحظي", layout="centered")
st.title("📊 جلب السعر اللحظي من الأسواق")

api_key = "cvtcvi1r01qhup0vnjrgcvtcvi1r01qhup0vnjs0"

market = st.selectbox("🗂️ اختر السوق:", ["🇺🇸 السوق الأمريكي", "🏦 السوق السعودي", "₿ العملات الرقمية"])
user_input = st.text_input("🔍 أدخل رمز السهم أو العملة:", "AAPL")

if st.button("📥 جلب السعر اللحظي"):
    symbol = user_input.upper()

    # السوق السعودي (yfinance)
    if market == "🏦 السوق السعودي":
        ticker = symbol + ".SR"
        try:
            df = yf.download(ticker, period="5d", interval="15m")
            if not df.empty and "Close" in df.columns:
                last_price = df["Close"].dropna().iloc[-1]
                st.success(f"✅ السعر اللحظي لـ {symbol}: {last_price:.2f} ريال")
            else:
                st.warning("⚠️ لا توجد بيانات حديثة.")
        except Exception as e:
            st.error(f"❌ خطأ أثناء تحميل البيانات: {e}")

    # السوق الأمريكي (Finnhub)
    elif market == "🇺🇸 السوق الأمريكي":
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
        response = requests.get(url)
        data = response.json()
        if "c" in data and data["c"]:
            price = float(data["c"])
            st.success(f"✅ السعر اللحظي لـ {symbol}: {price:.2f} دولار")
        else:
            st.warning("⚠️ لم يتم العثور على السعر.")

    # العملات الرقمية (Finnhub)
    elif market == "₿ العملات الرقمية":
        url = f"https://finnhub.io/api/v1/quote?symbol=BINANCE:{symbol}USDT&token={api_key}"
        response = requests.get(url)
        data = response.json()
        if "c" in data and data["c"]:
            price = float(data["c"])
            st.success(f"✅ السعر اللحظي لـ {symbol}: {price:.2f} دولار")
        else:
            st.warning("⚠️ تعذر العثور على بيانات العملة الرقمية.")
