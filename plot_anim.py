"""import matplotlib.pyplot as plt
import random
values = [0] * 50

for i in range(50):
    values[i] = random.randint(0, 100)
    #print(values[i])
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.bar(list(range(50)), values)
    plt.pause(0.00001)

plt.show()"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

fig, ax = plt.subplots()

max_x = 5
max_rand = 10

x = np.arange(0, max_x)
ax.set_ylim(0, max_rand)
line, = ax.plot(x, np.random.randint(0, max_rand, max_x))
the_plot = st.pyplot(plt)

def init():  # give a clean slate to start
    line.set_ydata([np.nan] * len(x))

def animate(i):  # update the y values (every 1000ms)
    line.set_ydata(np.random.randint(0, max_rand, max_x))
    the_plot.pyplot(plt)

init()
for i in range(100):
    animate(i)
    time.sleep(0.1)