from __future__ import division
import cv2
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadImages(path = "C:/Users/Mynha/Desktop/Dataset/batanes/"):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]


cols = ['pixel_w','varieties']
df = pd.DataFrame([], columns=cols)

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
		if (area >= 100000 and area <= 900000): # (4-9, top) (1-9, bot,sides)
			contour_list.append(contour)
			lst_bgr = []
			lst_hsv = []
			lst_gray = []
			lst_w = []
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

				# To confirm exact locations
				# print 'height', height
				# print 'width', width
				# np.savetxt('myfile2.csv', zip(height,width), delimiter=',', fmt='%.18g')
				# plt.imshow(mask)
				# plt.show()

				## Grayscale
				lst_gray.append(gray[height,width])
				arr_gray = lst_gray[i]
				garlic = len(arr_gray)

				## RGB
				lst_bgr.append(a[height,width])
				arr_bgr = lst_bgr[i]
				## HSV
				lst_hsv.append(hsv[height,width])
				arr_hsv = lst_hsv[i]

				# arr -> list of the pixel values of the garlic object

				# Color Based Features
				# Color Mean and Standard Deviation
				## RGB
				B, G, R = arr_bgr[:,0], arr_bgr[:,1], arr_bgr[:,2]
				mean_b = np.mean(B).flatten()
				mean_g = np.mean(G).flatten()
				mean_r = np.mean(R).flatten()
				std_b = np.std(B).flatten()
				std_g = np.std(G).flatten()
				std_r = np.std(R).flatten()
				## HSV
				H, S, V = arr_hsv[:,0], arr_hsv[:,1], arr_hsv[:,2]
				mean_h = np.mean(H).flatten()
				mean_s = np.mean(S).flatten()
				mean_v = np.mean(V).flatten()
				std_h = np.std(H).flatten()
				std_s = np.std(S).flatten()
				std_v = np.std(V).flatten()
				# Raw Feature Vector
				raw_b = B.flatten()
				raw_g = G.flatten()
				raw_r = R.flatten()
				## HSV
				raw_h = H.flatten()
				raw_s = S.flatten()
				raw_v = V.flatten()

				## Whiteness - BGR
				boundaries = [
					([217, 217, 217], [255, 255, 255])
				]
				for (lower, upper) in boundaries:
					lower = np.array(lower, dtype = "uint8")
					upper = np.array(upper, dtype = "uint8")

					mask_w = cv2.inRange(mask, lower, upper)


				# plt.imshow(mask_w)
				# plt.show()
				## Get White Pixels from Garlic Object
				index = np.where(mask_w == 255)
				h = index[0]
				w = index[1]

				lst_w.append(gray[h,w])
				arr_w = lst_w[i]
				whiteness = len(arr_w)
				# whiteness = list(arr_w)
				# whiteness = len(whiteness)
					

					# np.savetxt('white.csv', arr_w, delimiter=',', fmt='%.18g')

				# print 'whole garlic', garlic
				# print 'white pixels', whiteness

				pixel_w = whiteness / garlic
				print pixel_w
				# print "Whiteness:", type(pixel_w)
				# plt.imshow(mask_w, cmap='gray')
				# plt.show()


				# Save to CSV File
				# varieties = '0'
				# dict = {'pixel_w': pixel_w, 'varieties' : varieties}
				# df_temp = pd.DataFrame(dict)
				# df = df.append(df_temp, sort=False, ignore_index=True)
				# df.to_csv('color_features1.csv')


cv2.waitKey(0)
cv2.destroyAllWindows()