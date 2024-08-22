import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./Lab1/image.jpeg')
plt.subplot(2, 3, 1)
imgplot = plt.imshow(img)

gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
gray.astype(np.uint8)
plt.subplot(2, 3, 2)
imgplot = plt.imshow(gray, cmap='gray')

neg = 255 - gray
plt.subplot(2, 3, 3)
imgplot = plt.imshow(neg, cmap='gray')

plt.show()