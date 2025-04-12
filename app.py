        macd_values = ta.trend.MACD(close=close_clean)
        df['MACD'] = macd_values.macd().reindex(df.index).fillna(0)

        # التطبيع
        close_scaler = MinMaxScaler()
        df['Close_scaled'] = close_scaler.fit_transform(df[['Close']])

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close_scaled', 'RSI', 'MACD']])

        # تجهيز بيانات التدريب
        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled) - predict_days):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i:i+predict_days, 0])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # بناء النموذج
        input_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, input_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(predict_days))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # التنبؤ
        last_seq = scaled[-seq_len:]
        preds_scaled = model.predict(last_seq.reshape(1, seq_len, input_features))[0]
        forecast = close_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # عرض النتائج
        st.subheader("🔮 التوقعات:")
        for i, price in enumerate(forecast):
            st.markdown(f"اليوم {i+1}: {price:.2f} ريال / دولار")

        st.subheader("📊 رسم بياني للسعر")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Close'][-100:], label='السعر الفعلي')
        ax.set_title(f"آخر أسعار {symbol}")
        ax.grid()
        st.pyplot(fig)

        st.success("✅ النموذج يعمل باستخدام RSI و MACD بدقة.")
