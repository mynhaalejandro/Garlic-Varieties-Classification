import cv2
import os
import csv
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
	clone1 = a.copy()

	## Convert to Grayscale, HSV, Binary using Watershed Algo
	gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	## Noise removal
	kernel = np.ones((5,5),np.uint8) # (5, top) (3, bot,sides)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
	## Sure background area
	sure_bg = cv2.dilate(closing, kernel, iterations=3)

	## Get contour points		
	contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if (area >= 400000 and area <= 900000): # (4-9, top) (1-9, bot,sides)
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

				## Append all Image pixels
				indices = np.where(cimg == 255)
				height = indices[0]
				width = indices[1]
				lst_intensities.append(a[height,width])
				arr = lst_intensities[i]

				# Color Based Features
				## BGR Channel - 256 values

				# Raw Pixel Feature Vectors 
				## A list of numbers corresponding to the raw RGB pixel intensities of my image
				# raw_rgb = arr.flatten()
				# print raw_rgb.shape
				# print raw_rgb

				# Color Mean and Standard Deviation
				## The mean and standard deviation value of each channel of the image
				# B, G, R = arr[:,0], arr[:,1], arr[:,2]
				# mean_b = np.mean(B).flatten()
				# mean_g = np.mean(G).flatten()
				# mean_r = np.mean(R).flatten()
				# std_b = np.std(B).flatten()
				# std_g = np.std(G).flatten()
				# std_r = np.std(R).flatten()

				# 1D Color Histograms
				# histB = cv2.calcHist([clone1[..., 0]], [0], cimg, [256], [0, 256])  # im[...,0] is the Blue channel
				# histG = cv2.calcHist([clone1[..., 1]], [0], cimg, [256], [0, 256])  # im[...,1] is the Green channel
				# histR = cv2.calcHist([clone1[..., 2]], [0], cimg, [256], [0, 256])  # im[...,2] is the Red channel 
				# plt.plot(histB, color='b') 
				# plt.plot(histG, color='g')
				# plt.plot(histR, color='r')
				# plt.xlim([0, 256])
				# plt.title('Distribution of RGB in the Segmented Image')
				# plt.xlabel('Pixel Intensity')
				# plt.ylabel('Quantity')
				# plt.show()

				# 3D Color Histograms
				## A list of numbers used to characterize the color distribution of the image
				# rgb_hist = cv2.calcHist([clone1], [0, 1, 2], cimg, [8, 8, 8], [0, 256, 0, 256, 0, 256])
				# rgb_hist = cv2.normalize(rgb_hist)
				# rgb_hist = rgb_hist.flatten()
				# print '3D RGB', rgb_hist

				# Find the peak values for R, G, and B
				# chans = cv2.split(clone1)
				# colors = ('b', 'g', 'r')
				# rgb_features = []
				# feature_data = ''
				# counter = 0
				# for (chan, color) in zip(chans, colors):
				# 	counter = counter + 1

				# 	hist = cv2.calcHist([chan], [0], cimg, [256], [0,256])
				# 	rgb_features.extend(hist)

				# 	# find the peak pixel values for R, G, and B
				# 	elem = np.argmax(hist)

				# 	if counter == 1:
				# 		blue = str(elem)
				# 	elif counter == 2:
				# 		green = str(elem)
				# 	elif counter == 3:
				# 		red = str(elem)
				# 		feature_data = red + ',' + green + ',' + blue
					
				# print feature_data
				# with open("feature_data.csv", "a") as myfile:
				# 	myfile.write(feature_data + '\n')

				# Normalized difference index in RGB

				## Save PNG image using opencv
				# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/thresholded_bot_{}.png".format(len(images)+1), cimg)
				# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/cropped_bot_{}.png".format(len(images)+1), mask)
				## Show plot using matplotlib
				# plt.imshow(hsv)
				# plt.imshow(cimg, cmap='gray')
				# plt.imshow(mask, cmap='gray')
				# plt.show()


	images.append(stats_R)
# We use Boxplot, to vizualize the comparison of the different garlic varieties
batanes = images[0]
ilocos_pink = images[1]
ilocos_white = images[2]
mexican = images[3]
mmsu_gem = images[4]
tanbolters = images[5]
vfta = images[6]

fig = plt.figure(2)
fig.suptitle('RGB Color Space', fontsize=14, fontweight='bold')
plt.xlabel("Different Garlic Varieties")
plt.ylabel("Pixel Intensity")
plt.boxplot([batanes, ilocos_pink, ilocos_white, mexican, mmsu_gem, tanbolters, vfta], 
	labels=['batanes','ilocos_pink','ilocos_white','mexican', 'mmsu_gem', 'tanbolters', 'vfta'])
fig.savefig('RGB Color Space.png',dpi=100)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()