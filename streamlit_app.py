import streamlit as st
import pandas as pd
import plost

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time
import matplotlib.animation as animation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import chart_studio.plotly as py
import plotly.express as px

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `version 2`')

st.sidebar.subheader('Heat map parameter')
time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.subheader('ESP chart parameters')
plot_data2 = st.sidebar.multiselect('Select data', ['disch_pres', 'avg_amps'], ['disch_pres', 'avg_amps'])
#plot_height2 = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [Data Professor](https://youtube.com/dataprofessor/).
''')


# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
#for i in range(0, 10)

col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")
col4.metric("Temperature", "70 °F", "1.2 °F")
col5.metric("Wind", "9 mph", "-8%")
col6.metric("Humidity", "86%", "4%")

# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns((5,5))
with c1:
    st.markdown('### Heatmap')
    plost.time_hist(
    data=seattle_weather,
    date='date',
    x_unit='week',
    y_unit='day',
    color=time_hist_color,
    aggregate='median',
    legend=None,
    height=345,
    use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)


# Row D
esp = pd.read_csv("esppetang.csv")
esp.timestamp = pd.to_datetime(esp.timestamp)
esp
st.markdown('### ESP Line chart')
st.line_chart(esp, x = 'timestamp', y = plot_data2)

# Row E
"""import random
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
%matplotlib qt

l1 = [random.randint(-10,4)+(i**1.68)/(random.randint(13,14)) for i in range(0,160,2)]
l2 = [random.randint(0,4)+(i**1.5)/(random.randint(9,11)) for i in range(0,160,2)]
l3 = [random.randint(-10,10)-(i**1.3)/(random.randint(11,12)) for i in range(0,160,2)]

from itertools import count
xval= count(0,3)

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,5))
axes.set_ylim(-100, 500)
axes.set_xlim(0, 250)
plt.style.use("ggplot")

x1,y1,y2,y3 = [], [], [], []
xval= count(0,3)
def animate(i):
    x1.append(next(xval))
    y1.append((l1[i]))
    y2.append((l2[i]))
    y3.append((l3[i]))

    axes.plot(x1,y1, color="red")
    axes.plot(x1,y2, color="gray", linewidth=0.5)
    axes.plot(x1,y3, color="blue")
    
anim = FuncAnimation(fig, animate, interval=30)"""
