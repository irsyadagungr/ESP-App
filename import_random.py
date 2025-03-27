import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(10,5))
x, y = [], []

def animate(i):
    x.append(i)
    y.append(random.randint(0, 50))
    plt.style.use("ggplot")
    plt.plot(x, y)

ani = FuncAnimation(fig, animate, interval=300)
plt.show()