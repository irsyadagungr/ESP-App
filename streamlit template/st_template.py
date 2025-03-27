import streamlit as st

with st.sidebar:
    st.header("This is the header")
    st.subheader("This is the subheading")
    st.write(10+20)
st.header("Hello")

#columns
col1, col2 = st.columns(2)
with col1:
    st.text("Welcome")
    st.write("Subs to my channel")

with col2:
    st.text("Not Welcome")
    st.write("Subs to my channel")

"""col1.header("Hi")
col1.text("Okay")

col2.header("hello")"""