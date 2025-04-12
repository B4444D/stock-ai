/mnt/data/app-enhanced.py

        # مقارنة التوقعات السابقة (المرحلة 2)
        st.subheader("📋 مراجعة التوقعات السابقة")
        import glob
        review_files = glob.glob("forecasts/forecast_*.csv")
        review_results = []

        for file in review_files:
            try:
                df_forecast = pd.read_csv(file)
                forecast_dates = pd.to_datetime(df_forecast['date'])
                predicted = df_forecast['predicted_close']

                # جلب الأسعار الحقيقية من yfinance
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
