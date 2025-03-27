import streamlit as st
import pandas as pd
import plotly.express as px

with st.sidebar:
    st.header("This is the header")
    st.subheader("This is the subheading")
    st.write(10+20)
st.header("Hello")

df = pd.read_csv("esppetang.csv")
print(df)

fig = px.line(df, x="timestamp", y=df["Discharge Pressure(psi)"], title='Life expectancy in Canada')

#columns
col1, col2 = st.columns(2)


col1.header("Hi")
col1.text("Okay")

col2.header("hello")
#fig.show()