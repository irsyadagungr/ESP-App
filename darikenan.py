import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import datetime
import numpy as np
from ipywidgets import widgets

import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/SLB_Logo_2022.svg/640px-SLB_Logo_2022.svg.png",
    layout="wide",
)

# read csv from a github repo
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"
esp = pd.read_csv("esppetang.csv")
esp.timestamp = pd.to_datetime(esp.timestamp)

# read csv from a URL
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

st.sidebar.subheader('ESP chart parameters')
plot_data2 = st.sidebar.multiselect('Select data', ['disch_pres', 'avg_amps'], ['disch_pres', 'avg_amps'])

# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# top-level filters
job_filter = st.selectbox("Select the Job", pd.unique(df["job"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df = df[df["job"] == job_filter]

h = 0
m = 1

# near real-time / live feed simulation
for seconds in range(200):
    df["age_new"] = df["age"] * np.random.choice(range(1, 5))
    df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))

    #h += 1
    m += 1

    # creating KPIs
    avg_age = np.mean(df["age_new"])

    count_married = int(
        df[(df["marital"] == "married")]["marital"].count()
        + np.random.choice(range(1, 30))
    )

    balance = np.mean(df["balance_new"])

    

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Age ⏳",
            value=round(avg_age),
            delta=round(avg_age) - 10,
        )
        
        kpi2.metric(
            label="Married Count 💍",
            value=int(count_married),
            delta=-10 + count_married,
        )
        
        kpi3.metric(
            label="A/C Balance ＄",
            value=f"$ {round(balance,2)} ",
            delta=-round(balance / count_married) * 100,
        )

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Age ⏳",
            value=round(h),
            delta=round(h) - 10,
        )
        
        kpi2.metric(
            label="Married Count 💍",
            value=int(count_married),
            delta=-10 + count_married,
        )
        
        kpi3.metric(
            label="A/C Balance ＄",
            value=round(m),
            delta=round(m) - 10,
        )

        st.markdown("### First Chart")
        fig = px.density_heatmap(
            data_frame=df, y="age_new", x="marital"
        )
        st.write(fig)

        st.markdown("# ESP DATA")
        fig = (px.line(data_frame=esp, y=esp["avg_amps"][h:m], x=esp["timestamp"][h:m]),
            px.line(data_frame=esp, y=esp["Intake Pressure(psi)"][h:m], x=esp["timestamp"][h:m]),
            px.line(data_frame=esp, y=esp["disch_pres"][h:m], x=esp["timestamp"][h:m]),
            px.line(data_frame=esp, y=esp["Intake Temperature(F)"][h:m], x=esp["timestamp"][h:m]),
            px.line(data_frame=esp, y=esp["Drive Frequency(Hz)"][h:m], x=esp["timestamp"][h:m]),
            px.line(data_frame=esp, y=esp["Motor Temperature(F)"][h:m], x=esp["timestamp"][h:m]),
        )
        st.write(fig)
        esp["avg_amps"][0:10]
        
        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.density_heatmap(
                data_frame=df, y="age_new", x="marital"
            )
            st.write(fig)
            
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame=df, x="age_new")
            st.write(fig2)

        st.markdown("### Detailed Data View")
        st.dataframe(df)
        time.sleep(1)