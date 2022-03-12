import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-40., 40.)

y = (x - 30.) * 0.1 * (x - 30. > 0.)
z = -0.1 * x + 2.5
z = np.clip(z, 0.0, 1.0)

plt.plot(x, y)
plt.plot(x, z)
plt.show()