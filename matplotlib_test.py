import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100) * np.random.normal(50, 3)
y = np.random.randn(100) * np.random.normal(-40, 3)
z = np.random.randn(100) * np.random.normal(500, 3)

fig = plt.figure(figsize=(20, 16), dpi=70)

ax1 = fig.add_subplot(2, 2, 1)
rng = np.arange(len(x))
plt.plot(rng, x, "bs")  # ro = point, bs = squares

ax2 = fig.add_subplot(2, 2, 2)
rng = np.arange(len(y))
plt.axis([0, 50, -100, 100])
plt.plot(rng, y, "r--")  # r- solid, r-- = discontinue

ax3 = fig.add_subplot(2, 2, 3)
rng = np.arange(len(z))
plt.plot(rng, z, "g^")

ax4 = fig.add_subplot(2, 2, 4)
labels = ['perro', 'gato', 'sapo', 'pinguino']
sizes = [20, 45, 15, 45]
explode = [0.2, 0.0, 0.0, 0.0]
ax4.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.0f%%')
ax4.axis('equal')

plt.tight_layout()
plt.show()
plt.close(fig)
