import numpy as np
import matplotlib.pyplot as plt



img = np.zeros((10,10))
img[5,5] = 1

plt.imshow(img, cmap='gray')
plt.show()