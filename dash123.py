import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

dif = pd.read_csv("esppetangkenan2.csv")
dif.drop(columns=['Gas Lock', 'Dif'], inplace=True)
dif = dif[dif.State != "CLEAR"]


# Create the plot
fig = go.Figure()

# Add your data traces for the y1 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Discharge Pressure(psi)"], mode="lines", name="y1"))

# Add your data traces for the y2 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Average Amps(Amps)"], mode="lines", name="y2", yaxis="y2"))

# Add your data traces for the y3 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Intake Temperature(F)"], mode="lines", name="y3", yaxis="y3"))

# Add your data traces for the y4 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Drive Frequency(Hz)"], mode="lines", name="y4", yaxis="y4"))

# Add your data traces for the y5 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Motor Temperature(F)"], mode="lines", name="y5", yaxis="y5"))

# Add your data traces for the y6 axis
fig.add_trace(go.Scatter(x=dif["timestamp"], y=dif["Intake Pressure(psi)"], mode="lines", name="y6", yaxis="y6"))

# Add the secondary y-axes
fig.update_layout(
    yaxis2=dict(overlaying='y', side='right'),
    yaxis3=dict(overlaying='y', side='right'),
    yaxis4=dict(overlaying='y', side='left'),
    yaxis5=dict(overlaying='y', side='left'),
    yaxis6=dict(overlaying='y', side='left'),
)

# Add the dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list(dif["timestamp"]), 
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0.1,
            xanchor='left',
            y=1.1,
            yanchor='top'
        )]
)
# Display the plot
fig.show()