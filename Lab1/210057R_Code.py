import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./Lab1/210057R_SrcImage.jpg')
if img is None:
    print("Image not found.")
    exit()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

gray_8 = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
gray_8.astype(np.uint8)
axes[0,0].imshow(gray_8, cmap='gray')
axes[0,0].set_title('Unprocessed Grayscale')


negative = 255 - gray_8
negative = negative.astype(np.uint8)
axes[0,1].imshow(negative, cmap='gray')
axes[0,1].set_title('Negative')


increased_brightness = np.clip(gray_8.astype(np.int16)*1.2,0,255).astype(np.uint8)
axes[0,2].imshow(increased_brightness, cmap='gray',vmin=0, vmax=255)
axes[0,2].set_title('Brightness + 20%')


low_contrast = np.interp(gray_8, (0, 255), (125, 175))
low_contrast = low_contrast.astype(np.uint8)
axes[1,0].imshow(low_contrast, cmap='gray',vmin=0, vmax=255)
axes[1,0].set_title('Low Contrast')


low_depth = gray_8 // 16 * 16
low_depth = low_depth.astype(np.uint8)
axes[1,1].imshow(low_depth, cmap='gray',vmin=0, vmax=255)
axes[1,1].set_title('4bpp')


flipped = [np.flip(array) for array in gray_8]
flipped = np.array(flipped).astype(np.uint8)
axes[1,2].imshow(flipped, cmap='gray',vmin=0, vmax=255)
axes[1,2].set_title('Vertical Mirror')

for ax in axes.flat:
    ax.axis('off')

plt.savefig('./Lab1/210057R_SubPlot.png')

cv2.imwrite('./Lab1/210057R_OPImage_11.jpg', gray_8)
cv2.imwrite('./Lab1/210057R_OPImage_12.jpg', negative)
cv2.imwrite('./Lab1/210057R_OPImage_13.jpg', increased_brightness)
cv2.imwrite('./Lab1/210057R_OPImage_21.jpg', low_contrast)
cv2.imwrite('./Lab1/210057R_OPImage_22.jpg', low_depth)
cv2.imwrite('./Lab1/210057R_OPImage_23.jpg', flipped)

plt.show()



