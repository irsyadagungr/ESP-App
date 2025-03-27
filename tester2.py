import streamlit as st
import plotly.express as px
import plotly.subplots as subplots
import pandas as pd

# Load the data into a Pandas DataFrame
df = pd.read_csv('data.csv')

# Create a subplot with two y-axes
fig = subplots.subplots(specs=[[{"secondary_y": True}, {"secondary_y": True}]])

# Add a trace to the subplot using the first y-axis
fig.add_trace(go.Scatter(x=df['x'], y=df['y1'], mode='lines', name='y1'))

# Add a trace to the subplot using the second y-axis
fig.add_trace(go.Scatter(x=df['x'], y=df['y2'], mode='lines', name='y2', yaxis='y2'))

# Update the layout of the plot
fig.update_layout(
  title='Multiple Y-Axes Example',
  width=800,
  height=600
)

# Display the plot
st.plotly_chart(fig)
