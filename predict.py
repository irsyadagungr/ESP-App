import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

print("a")

esp = pd.read_csv("esppetang2.csv")
espc = esp.copy()

print(espc)

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
    return Xs

#SCALER
scale_columns = ["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
                "Average Amps(Amps)", "Intake Temperature(F)", "Drive Frequency(Hz)"]

scaler = RobustScaler()

scaler = scaler.fit(esp[scale_columns])

esp.loc[:, scale_columns] = scaler.transform(
  esp[scale_columns].to_numpy()
)

#STEPS BARU
TIME_STEPS = 11
STEP = 11

X = create_dataset(
    esp[["Motor Temperature(F)", "Intake Pressure(psi)", "Discharge Pressure(psi)",
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

espc["State"] = zurface
espc