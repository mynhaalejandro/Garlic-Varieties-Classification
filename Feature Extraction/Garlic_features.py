import cv2
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadImages(path = "C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Images/sides/"):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]


cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b','mean_h','mean_s','mean_v','std_h','std_s','std_v','varieties']
df = pd.DataFrame([], columns=cols)

filenames = loadImages()
images = []
for file in filenames:
	a = cv2.imread(file)
	print(file)

	clone = a.copy()

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
				## Grayscale
				lst_gray.append(gray[height,width])
				arr_gray = lst_gray[i]
				## RGB
				lst_bgr.append(a[height,width])
				arr_bgr = lst_bgr[i]
				## HSV
				lst_hsv.append(hsv[height,width])
				arr_hsv = lst_hsv[i]

				# print 'height', height
				# print 'width', width
				# plt.imshow(a)
				# plt.show()

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

				# Save to CSV File
				varieties = '0'
				dict = {'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b, 'std_r': std_r, 'std_g': std_g, 'std_b': std_b, 'mean_h': mean_h, 'mean_s': mean_s, 'mean_v': mean_v, 'std_h': std_h, 'std_s': std_s, 'std_v': std_v, 'varieties' : varieties}
				df_temp = pd.DataFrame(dict)
				df = df.append(df_temp, sort=False, ignore_index=True)
				# df.to_csv('color_features1.csv')
	images.append(mean_h)

# We use Boxplot, to vizualize the comparison of the different garlic varieties
## By Single Image
batanes = images[0]
ilocos_pink = images[1]
ilocos_white = images[2]
mexican = images[3]
mmsu_gem = images[4]
tanbolters = images[5]
vfta = images[6]

fig = plt.figure(1)
fig.suptitle('Mean HUE', fontsize=14, fontweight='bold')
plt.xlabel("Different Garlic Varieties")
plt.ylabel("Pixel Intensity")
plt.boxplot([batanes, ilocos_pink, ilocos_white, mexican, mmsu_gem, tanbolters, vfta], 
	labels=['batanes','ilocos_pink','ilocos_white','mexican', 'mmsu_gem', 'tanbolters', 'vfta'])
# fig.savefig('Mean_Hue.png',dpi=100)
plt.show()

## By Multiple Images - by appending the array of the images into one list

cv2.waitKey(0)
cv2.destroyAllWindows()