import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

def loadImages(path = "C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Images/sides/"):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

filenames = loadImages()
images = []
for file in filenames:
	a = cv2.imread(file)
	print(file)

	clone = a.copy()

	## Convert to Grayscale, Binary using Global Thresholding
	gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
	ret, threshold = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
				
	## Noise removal
	box_filtered_image = cv2.boxFilter(threshold, 0, (7, 7), threshold, (-1, -1), False, cv2.BORDER_DEFAULT)

	## Get Contour Points		
	contours, hierarchy = cv2.findContours(box_filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if (area >= 300000 and area <= 900000):
			contour_list.append(contour)
			lst_intensities = []
			cv2.drawContours(clone, contour_list, -1, (0, 255, 0), 2)

			for i in range(len(contour_list)):
				## Thresholded Image(Binary)
				cimg = np.zeros((clone.shape[0],clone.shape[1]), np.uint8)    
				cv2.drawContours(cimg, contour_list, i, color=255, thickness=-1)
				## Cropped Image
				mask_rgb1 = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
				# perform bitwise and on mask to obtain cut-out image that is not blue
				masked_upstate = cv2.bitwise_and(clone, mask_rgb1)
				# replace the cut-out parts with white
				# masked_replace_white = cv2.addWeighted(masked_upstate, 1, a, 1, 0)


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
				print arr

				# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/thresholded_bot_{}.png".format(len(images)+1), cimg)
				# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/cropped_bot_{}.png".format(len(images)+1), mask)

				plt.imshow(cimg, cmap='gray')
				plt.show()

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