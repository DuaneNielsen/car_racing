import numpy as np
from matplotlib import pyplot as plt

x, y = np.meshgrid(np.arange(10), np.arange(10))
x -= 5
y -= 5

z = x ** 2 + y ** 2

plt.imshow(10 - z)
plt.show()