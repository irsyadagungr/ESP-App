import plotly.graph_objects as go
import plotly.express as px

fig = go.Figure()
df = px.data.wind()
fig.add_trace(go.Scatter(
    x= df["direction"],
    y= df["frequency"]
))
fig.update_layout(
    title="Wind Frequencies",
    xaxis_title="Direction",
    yaxis_title="Frequency",
    autosize=False,
    width=6000,
    height=400,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
fig.show()