import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import datetime
import numpy as np
from ipywidgets import widgets
import plotly.graph_objects as go

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

import os
import streamlit as st

import os
import tarfile
import tensorflow as tf
import requests
from tensorflow.keras.layers import LSTM 

st.set_page_config(
    page_title="ESP Gas Lock Detection",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/SLB_Logo_2022.svg/640px-SLB_Logo_2022.svg.png",
    layout="wide",
)

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.subheader("Run the Program")
result = st.sidebar.button("Click Here")
st.write("The Program Can Run With or Without a CSV Upload")


if uploaded_file is not None and result:
    esp3 = pd.read_csv(uploaded_file)
    esp = esp3.copy()
    espj = esp3.copy()
    espc = esp3.copy()
    espd = esp3.copy()
else:
    #esp3 = pd.read_csv("espjanuari.csv")
    esp3 = pd.read_csv("esphelp1.csv")
    #esp3 = pd.read_csv(r"C:\Users\lenovo\PycharmProjects\SLBMM\Streamlit\esphelp1.csv")
    #bb = pd.read_csv("esppetang3.csv")
    esp = esp3.copy()
    espj = esp3.copy()
    espc = esp3.copy()
    espd = esp3.copy()



if result:
    #Data Awal
    esp2 = esp.copy() #Penampil Waktu Box
    esp4 = esp2.copy()
    #esp.timestamp = pd.to_datetime(esp.timestamp)

    # Convert 'timestamp' column to datetime format, handling errors
    esp["timestamp"] = pd.to_datetime(esp["timestamp"], errors="coerce")
    
    # Check for missing timestamps
    if esp["timestamp"].isnull().sum() > 0:
        print("Warning: Some timestamps could not be converted and are set to NaT")


    #Data Pembagian
    espn = esp2.iloc[0:1, :]
    espg = esp2.iloc[0:1, :]
    
    #with open('style.css') as f:
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Get absolute path of style.css
    css_path = os.path.join(os.path.dirname(__file__), "style.css")

    # Check if the file exists before opening
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"❌ File Not Found: {css_path}")  # Show an error in Streamlit

    # read csv from a URL


    # dashboard title
    st.title("ESP Gas Lock Detection Dashboard")

    # creating a single-element container
    placeholder = st.empty()

    #MODEL INITIALIZE
    #model_path = "my_models.h5" 
    #new_model = tf.keras.models.load_model(model_path)

    #tf.keras.utils.get_custom_objects().update({"LSTM": LSTM})
    #model_path = "my_models_fixed.h5"
    #new_model = tf.keras.models.load_model(model_path)

    # ✅ Define and register the custom LSTM
    @tf.keras.utils.register_keras_serializable()
    class CustomLSTM(LSTM):
        def __init__(self, *args, time_major=False, **kwargs):  # Remove `time_major`
            kwargs.pop("time_major", None)  # ✅ Remove the invalid argument
            super().__init__(*args, **kwargs)  # ✅ Pass the rest to the original LSTM
    
    # ✅ Register the class before loading the model
    tf.keras.utils.get_custom_objects().update({"CustomLSTM": CustomLSTM})
    
    # ✅ Load the model
    model_path = "my_models_fixed.h5"
    new_model = tf.keras.models.load_model(model_path)
    
    print("✅ Model loaded successfully!")

    #DATASET BARU
    def create_dataset(X, time_steps=1, step=1):
        Xs = []
        for i in range(0, len(X), step):
        #for i in range(0, 1881 - 11, step):
            v = X.iloc[i:(i + time_steps)].values
            #v = X.iloc[0 : 0 + 11].values
            #print(labels)
            #labels = y.iloc[0 : 0 + 11]
            Xs.append(v)
        return np.array(Xs)
    

    #near real-time / live feed simulation
    x = int(len(esp3)/11)
    xx = x
    a = 0; b = 11
    h = 1; m = 2
    d = 1; e = 2; f = 2
    q = 0
    espc = espc.iloc[a:b*x,:]
    zurface = []
    yurface = []

    
    
    for i in range(0, xx):
        #esp3 = pd.read_csv("espjanuari.csv")
        esp3 = pd.read_csv("esphelp1.csv")
        #esp3 = pd.read_csv(r"C:\Users\lenovo\PycharmProjects\SLBMM\Streamlit\esphelp1.csv")
        esp4 = esp3.iloc[0:b,:]
        espj = esp3.iloc[a:b,:]
        espjc = espj.copy()
        espj = pd.concat([espd, espj])

        #MODEL
        #SCALER
        scale_columns = ["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
                        "Average Amps(Amps)", "Intake Temperature(F)", "Drive Frequency(Hz)"]

        scaler = RobustScaler()

        scaler = scaler.fit(espj[scale_columns])

        espj.loc[:, scale_columns] = scaler.transform(
        espj[scale_columns].to_numpy()
        )

        #STEPS BARU
        TIME_STEPS = 11
        STEP = 11

        X = create_dataset(
            espj[["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
                        "Average Amps(Amps)", "Intake Temperature(F)", "Drive Frequency(Hz)"]],
            TIME_STEPS,
            STEP
        )

        #Encrypt
        #enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


        X = np.asarray(X).astype(np.float32)
        #tf.convert_to_tensor(X, dtype=tf.float32)

        y_pred = new_model.predict(X)

        y_pred = np.argmax(y_pred, axis=1)

        surface = []
        for ip in y_pred[-1:]:
            if ip == 0:
                surface.append("Gas Lock Detected")
            elif ip == 1:
                surface.append("Normal Condition")

        zurface = []
        for ix in surface[-1:]:
            for j in range(0, 11):
                zurface.append(ix)
        
        for iy in zurface:
            yurface.append(iy)
        
        esp4["State"] = yurface

        #dif["State"] = zurface
        espjc["State"] = zurface

        if zurface[-1] == "Gas Lock Detected":
            espg = pd.concat([espg, espjc])
            espg.timestamp = pd.to_datetime(espg.timestamp)

        elif zurface[-1] == "Normal Condition":
            espn = pd.concat([espn, espjc])
            espn.timestamp = pd.to_datetime(espn.timestamp)

        
        
        #espt = pd.concat(espt, espjc)
        
        num_data = 100
        timestamps = pd.date_range(start="2024-03-27 00:00:00", periods=num_data, freq="T")
        esp = pd.DataFrame({
            "timestamp": timestamps,
            "Discharge Pressure(psi)": np.random.randint(1000, 3000, num_data),
            "Average Amps(Amps)": np.random.randint(50, 150, num_data),
            "Intake Temperature(F)": np.random.randint(80, 120, num_data),
            "Drive Frequency(Hz)": np.random.randint(30, 60, num_data),
            "Motor Temperature(F)": np.random.randint(100, 200, num_data),
            "Intake Pressure(psi)": np.random.randint(50, 150, num_data)
        })
        
        # Initialize session state variables
        if "index" not in st.session_state:
            st.session_state.index = 0
        
        st.title("Real-Time ESP Monitoring")
        
        placeholder = st.empty()
        
        while True:
            m = st.session_state.index
        
            if m >= len(esp):
                st.session_state.index = 0  # Restart data loop
                continue
        
            with placeholder.container():
                st.markdown("### ESP Live Data")
        
                # Display KPIs
                col1, col2 = st.columns(2)
                col1.metric("Timestamp", esp["timestamp"][m])
                col2.metric("State", "Normal" if np.random.rand() > 0.5 else "Gas Lock Detected")  # Simulated state
        
                # Three KPI metrics
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Discharge Pressure (psi)", esp["Discharge Pressure(psi)"][m])
                kpi2.metric("Average Amps (Amp)", esp["Average Amps(Amps)"][m])
                kpi3.metric("Intake Temperature (F)", esp["Intake Temperature(F)"][m])
        
                # Additional KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Drive Frequency (Hz)", esp["Drive Frequency(Hz)"][m])
                kpi2.metric("Motor Temperature (F)", esp["Motor Temperature(F)"][m])
                kpi3.metric("Intake Pressure (psi)", esp["Intake Pressure(psi)"][m])
        
                # Line Chart for ESP Data
                st.markdown("### ESP Data Trends")
                fig = go.Figure()
        
                fig.add_trace(go.Scatter(x=esp["timestamp"][:m], y=esp["Discharge Pressure(psi)"][:m],
                                         mode="lines", name="Discharge Pressure", line_color="blue"))
        
                fig.add_trace(go.Scatter(x=esp["timestamp"][:m], y=esp["Average Amps(Amps)"][:m],
                                         mode="lines", name="Average Amps", line_color="red"))
        
                fig.add_trace(go.Scatter(x=esp["timestamp"][:m], y=esp["Intake Temperature(F)"][:m],
                                         mode="lines", name="Intake Temperature", line_color="green"))
        
                st.plotly_chart(fig, use_container_width=True)
        
                # Data Table
                st.markdown("### Data Table")
                st.dataframe(esp.iloc[:m])
        
            # Increment session state index
            st.session_state.index += 1
        
            # Sleep for real-time effect
            time.sleep(1)

        a+=11; b+=11
    
    esp4.to_csv("espdaribaca.csv")

