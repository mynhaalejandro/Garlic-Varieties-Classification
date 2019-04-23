import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

def loadImages(path = "C:/Users/Mynha/Desktop/Dataset/batanes/"):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

filenames = loadImages()
images = []
for file in filenames:
	a = cv2.imread(file)
	print(file)
	# print a.shape

	clone = a.copy()

	## Convert to Grayscale, Binary using Watershed Algo
	gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	## Noise removal
	kernel = np.ones((3,3),np.uint8) # 5, top 3, sides
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
	## Sure background area
	sure_bg = cv2.dilate(closing, kernel, iterations=3)

	## Get contour points		
	contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	plt.imshow(sure_bg)
	plt.show()
	contour_list = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if (area >= 100000 and area <= 900000): # 4-9, top 1-9, bot,sides
			contour_list.append(contour)
			lst_intensities = []
			cv2.drawContours(clone, contour_list, -1, (0, 255, 0), 2)

			for i in range(len(contour_list)):
				## Thresholded Image(Binary)
				cimg = np.zeros((clone.shape[0],clone.shape[1]), np.uint8)
				cv2.drawContours(cimg, contour_list, i, color=255, thickness=-1)
				## Cropped Image
				mask_rgb1 = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
				mask = cv2.bitwise_and(clone, mask_rgb1)

				## Replace Garlic to Blue Color
				# indices = np.where(cimg==255)
				# clone[indices[0], indices[1], :] = [0, 0, 255]

				## Append all Image pixels
				indices = np.where(cimg == 255)
				height = indices[0]
				width = indices[1]
				lst_intensities.append(gray[height,width])
				arr = lst_intensities[i]
				# Pixel values of the Garlic Object

				cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/thresholded_bot_{}.png".format(len(images)+1), cimg)
				cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/cropped_bot_{}.png".format(len(images)+1), mask)

				# plt.imshow(cimg, cmap='gray')
				# plt.show()

	images.append(arr)

## Pre-processing
# Visualization using Matplotlib
# fig = plt.figure(1)
# fig.suptitle('Segmentation', fontsize=14, fontweight='bold')
# ax = plt.subplot(2, 2, 1)
# ax.set_title('Orginal Image')
# plt.imshow(a)

# ax = plt.subplot(2, 2, 2)
# ax.set_title('Grayscale')
# plt.imshow(gray, cmap='gray')
# plt.colorbar()

# ax = plt.subplot(2, 2, 3)
# ax.set_title('Binary')
# plt.imshow(threshold, cmap='gray')        

# ax = plt.subplot(2, 2, 4)
# ax.set_title('Object Detected')
# plt.imshow(cimg, cmap='gray')
# plt.imsave('Thresholded Image Side.png', cimg, cmap='gray')
# print(cimg)
# plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()