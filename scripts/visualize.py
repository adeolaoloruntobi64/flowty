import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

DATA_FILE_MULTI = os.path.expanduser("~/pixel_training_dataset_multi.csv")

df = pd.read_csv(DATA_FILE_MULTI)

df = df[["Lab_L", "L_sl", "label"]]

fig = px.scatter(
    df,
    x="Lab_L",
    y="L_sl",
    #z="L_sl",
    color="label",
    hover_name=df.index
)

fig.update_traces(marker=dict(size=2))

fig.show()