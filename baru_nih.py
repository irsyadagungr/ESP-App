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

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

st.set_page_config(
    page_title="Real-Time ESP Dashboard",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/SLB_Logo_2022.svg/640px-SLB_Logo_2022.svg.png",
    layout="wide",
)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.subheader("Run the Program")
result = st.sidebar.button("Click Here")
st.write("The Program Can Run With or Without a CSV Upload")

if uploaded_file is not None and result:
    esp3 = pd.read_csv(uploaded_file)
    esp = esp3.copy()
else:
    esp3 = pd.read_csv("esppetang2.csv")
    esp = esp3.copy()



if result:
    #Model Prediction
    #DATASET BARU
    def create_dataset(X, time_steps=1, step=1):
        Xs, ys = [], []
        for i in range(0, len(X), step):
        #for i in range(0, 1881 - 11, step):
            v = X.iloc[i:(i + time_steps)].values
            #v = X.iloc[0 : 0 + 11].values
            #print(labels)
            #labels = y.iloc[0 : 0 + 11]
            Xs.append(v)
        return np.array(Xs)

    #SCALER
    scale_columns = ["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
                    "Average Amps(Amps)", "Intake Temperature(F)", "Drive Frequency(Hz)"]

    scaler = RobustScaler()

    scaler = scaler.fit(esp[scale_columns])

    esp3.loc[:, scale_columns] = scaler.transform(
    esp3[scale_columns].to_numpy()
    )

    #STEPS BARU
    TIME_STEPS = 11
    STEP = 11

    X = create_dataset(
        esp3[["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
                    "Average Amps(Amps)", "Intake Temperature(F)", "Drive Frequency(Hz)"]],
        TIME_STEPS,
        STEP
    )

    #Encrypt
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    new_model = tf.keras.models.load_model('saved_model/my_model')

    X = np.asarray(X).astype(np.float32)

    y_pred = new_model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    surface = []
    for i in y_pred:
        if i == 0:
            surface.append("Gas Lock Detected")
        elif i == 1:
            surface.append("Normal Condition")

    zurface = []
    for i in surface:
        for j in range(0, 11):
            zurface.append(i)

    esp["State"] = zurface

    #Data Awal
    esp2 = esp.copy()
    esp.timestamp = pd.to_datetime(esp.timestamp)

    st.sidebar.subheader('ESP chart parameters')
    img = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/SLB_Logo_2022.svg/640px-SLB_Logo_2022.svg.png",
    plot_data2 = st.sidebar.multiselect('Select data', ['Discharge Pressure(psi)', 'Average Amps(Amps)'], ['Discharge Pressure(psi)', 'Average Amps(Amps)'])

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # read csv from a URL


    # dashboard title
    st.title("Real-Time / Live ESP Dashboard")

    # creating a single-element container
    placeholder = st.empty()

    h = 1
    m = 2

    # near real-time / live feed simulation
    for seconds in range(len(esp)):
        #h += 1
        m += 1

        with placeholder.container():
            
            kpi1, kpi2 = st.columns(2)
            kpi1.metric(
                label="Timestamp",
                value=esp2["timestamp"][m]
            )

            kpi2.metric(
                label="State",
                value=esp["State"][m]
            )
            
            # create three columns
            kpi1, kpi2, kpi3 = st.columns(3)

            # fill in those three columns with respective metrics or KPIs
            kpi1.metric(
                label="Discharge Pressure (psi)",
                value=esp["Discharge Pressure(psi)"][m],
                delta=esp["Discharge Pressure(psi)"][m] - esp["Discharge Pressure(psi)"][m-1],
            )
            
            kpi2.metric(
                label="Average Amps (Amp)",
                value=esp["Average Amps(Amps)"][m],
                delta=esp["Average Amps(Amps)"][m] - esp["Average Amps(Amps)"][m-1],
            )
            
            kpi3.metric(
                label="Intake Temperature(F)",
                value=esp["Intake Temperature(F)"][m],
                delta=esp["Intake Temperature(F)"][m] - esp["Intake Temperature(F)"][m-1],
            )

            # create three columns
            kpi1, kpi2, kpi3 = st.columns(3)

            # fill in those three columns with respective metrics or KPIs
            kpi1.metric(
                label="Drive Frequency(Hz)",
                value=esp["Drive Frequency(Hz)"][m],
                delta=esp["Drive Frequency(Hz)"][m] - esp["Drive Frequency(Hz)"][m-1],
            )
            
            kpi2.metric(
                label="Motor Temperature(F)",
                value=esp["Motor Temperature(F)"][m],
                delta=esp["Motor Temperature(F)"][m] - esp["Motor Temperature(F)"][m-1],
            )
            
            kpi3.metric(
                label="Intake Pressure(psi)",
                value=esp["Intake Pressure(psi)"][m],
                delta=esp["Intake Pressure(psi)"][m] - esp["Intake Pressure(psi)"][m-1],
            )

            st.markdown("# ESP DATA")
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Discharge Pressure(psi)"][h:m],
                name="Discharge Pressure(psi)", mode="lines", line_color="#FF00EE"
            ))

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Average Amps(Amps)"][h:m],
                name="Average Amps(Amps)",
                yaxis="y2", mode="lines", line_color="#AB5959"
            ))

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Intake Temperature(F)"][h:m],
                name="Intake Temperature(F)",
                yaxis="y3", mode="lines", line_color="#060CBD"
            ))

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Drive Frequency(Hz)"][h:m],
                name="Drive Frequency(Hz)",
                yaxis="y4", mode="lines", line_color="#96A35C"
            ))

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Motor Temperature(F)"][h:m],
                name="Motor Temperature(F)",
                yaxis="y5", mode="lines", line_color="#006B33"
            ))

            fig.add_trace(go.Scatter(
                x=esp["timestamp"][h:m],
                y=esp["Intake Pressure(psi)"][h:m],
                name="Intake Pressure(psi)",
                yaxis="y6", mode="lines", line_color="#98FFFD"
            ))

            # Create axis objects
            fig.update_layout(
                xaxis=dict(
                    domain=[0.075, 0.95]
                ),
                autosize=False,
                width=3000,
                height=2500,
                yaxis=dict(
                    titlefont=dict(
                        color="#FF00EE"
                    ),
                    tickfont=dict(
                        color="#FF00EE"
                    )
                ),
                yaxis2=dict(
                    titlefont=dict(
                        color="#AB5959"
                    ),
                    tickfont=dict(
                        color="#AB5959"
                    ),
                    anchor="free",
                    overlaying="y",
                    side="left",
                    position=0.025
                ),
                yaxis3=dict(
                    titlefont=dict(
                        color="#060CBD"
                    ),
                    tickfont=dict(
                        color="#060CBD"
                    ),
                    anchor="free",
                    overlaying="y",
                    side="left",
                    position=0.045
                ),
                yaxis4=dict(
                    titlefont=dict(
                        color="#96A35C"
                    ),
                    tickfont=dict(
                        color="#96A35C"
                    ),
                    anchor="x",
                    overlaying="y",
                    side="right",
                ),
                yaxis5=dict(
                    titlefont=dict(
                        color="#006B33"
                    ),
                    tickfont=dict(
                        color="#006B33"
                    ),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position= 0.97
                ),
                yaxis6=dict(
                    titlefont=dict(
                        color="#98FFFD"
                    ),
                    tickfont=dict(
                        color="#98FFFD"
                    ),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.99
                )
            )

            # Update layout properties
            fig.update_layout(
                title_text="Gas Lock Feature",
                width=100000,
            )
            
            st.write(fig)
            
            # 3 COLUMNS 1
            # create three columns
            c1, c2, c3 = st.columns((10/3, 10/3, 10/3))
            with c1:
                st.markdown('### Disch Pressure(psi)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Discharge Pressure(psi)"][h:m],
                    name="Discharge Pressure(psi)", mode="lines", line_color="#FF00EE"
                ))

                st.write(fig)

            with c2:
                st.markdown('### Average Amps(Amps)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Average Amps(Amps)"][h:m],
                    name="Discharge Pressure(psi)", mode="lines", line_color="#AB5959"
                ))

                st.write(fig)
            
            with c3:
                st.markdown('### Intake Temperature(F)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Intake Temperature(F)"][h:m],
                    name="Intake Temperature(F)", mode="lines", line_color="#060CBD"
                ))

                st.write(fig)
            
            # 3 COLUMNS 2
            # create three columns
            c1, c2, c3 = st.columns((10/3, 10/3, 10/3))
            with c1:
                st.markdown('### Drive Frequency(Hz)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Drive Frequency(Hz)"][h:m],
                    name="Drive Frequency(Hz)", mode="lines", line_color="#96A35C"
                ))

                st.write(fig)

            with c2:
                st.markdown('### Motor Temperature(F)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Motor Temperature(F)"][h:m],
                    name="Motor Temperature(F)", mode="lines", line_color="#006B33"
                ))

                st.write(fig)
            
            with c3:
                st.markdown('### Intake Pressure(psi)')
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=esp["timestamp"][h:m],
                    y=esp["Intake Pressure(psi)"][h:m],
                    name="Intake Pressure(psi)", mode="lines", line_color="#98FFFD"
                ))

                st.write(fig)

            st.markdown("### Detailed Data View")
            st.dataframe(esp2[h:m])
            time.sleep(0.1)

