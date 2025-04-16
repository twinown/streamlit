import streamlit as st
import yfinance as yf


st.title('Данные о котировках компаниии Apple')

tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period = '1d',start = '2015-4-15',end = '2025-4-15')

st.write("""
         Закрытие торгов
         """)
st.line_chart(tickerDf.Close)
st.write("""
         Количество акций
         """)
st.line_chart(tickerDf.Volume)

