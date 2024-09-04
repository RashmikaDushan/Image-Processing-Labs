import cv2
import numpy as np

filename = "Lab2/source_images/road57.png" # Defining the file path

img = cv2.imread(filename)
if img is None:
    print("Image not found.")
    exit()

gray_8 = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
gray_8 = gray_8.astype(np.uint8)

low = gray_8.min()
high = gray_8.max()

original = (np.round(((gray_8-low)/(high-low))*255)).astype(np.uint8)
cv2.imwrite("Lab2/original.jpg", original) # Saving the original image

with_padding = original
padding = np.zeros(len(with_padding[0])).astype(np.uint8)

for i in range(2):
    with_padding = np.insert(with_padding, 0, 0, axis=0)
    with_padding = np.append(with_padding, [padding], axis=0)

for i in range(2):
    with_padding = np.insert(with_padding, 0, 0, axis=1)
    with_padding = np.insert(with_padding,with_padding.shape[1],0,axis=1)

filterA = np.array([ # Defining the filters
    [0, -1, -1, -1, 0],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [0, -1, -1, -1, 0]
])

filterB = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

filterC = np.array([
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5]
])

filterD = np.array([
    [0, -1, -1, -1, 0],
    [-1, 2, 2, 2, -1],
    [-1, 2, 16, 2, -1],
    [-1, 2, 2, 2, -1],
    [0, -1, -1, -1, 0]
])

filterA = filterA / np.sum(np.abs(filterA)) # Normalizing the filter
filterB = filterB / np.sum(np.abs(filterB))
filterC = filterC / np.sum(np.abs(filterC))
filterD = filterD / np.sum(np.abs(filterD))

def apply_filter(with_padding, filter): # Applying filter to the image
    newImage = np.zeros(original.shape)

    for i in range(len(newImage)):
        for j in range(len(newImage[0])):
            newImage[i][j] = np.sum(with_padding[i:i+len(filter),j:j+len(filter)]*filter) # Applying filter to the image

    return newImage # Returning the new image


imageA = apply_filter(with_padding, filterA) # Applying the filters to the image
imageB = apply_filter(with_padding, filterB)
imageC = apply_filter(with_padding, filterC)
imageD = apply_filter(with_padding, filterD)

cv2.imwrite("Lab2/filtered.jpg", imageA) # Saving the images
cv2.imwrite("Lab2/filtered2.jpg", imageB)
cv2.imwrite("Lab2/filtered3.jpg", imageC)
cv2.imwrite("Lab2/filtered4.jpg", imageD)

def rms(original, filtered): # Calculating the RMS value
    return np.sqrt(np.mean((original - filtered) ** 2))

print("RMS 1:",rms(original, imageA)) # Printing the RMS values
print("RMS 2:",rms(original, imageB))
print("RMS 3:",rms(original, imageC))
print("RMS 4:",rms(original, imageD))