import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

def loadImages(path = "C:/Users/Mynha/Desktop/"):
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
	blur = cv2.GaussianBlur(gray, (55,55),0)
	ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	## Noise removal
	kernel = np.ones((15,15),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
	## Sure background area
	sure_bg = cv2.dilate(closing, kernel, iterations=3)
	## Sure background area
	contours, hierarchy = cv2.findContours(sure_bg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	plt.imshow(sure_bg, cmap='gray')
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

				plt.imshow(cimg, cmap='gray')
				plt.show()

	# def find_contour(cnts):
	#     contains = []
	#     y_ri,x_ri, _ = a.shape
	#     for cc in cnts:
	#         yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
	#         contains.append(yn)

	#     val = [contains.index(temp) for temp in contains if temp>0]
	#     print(contains)
	#     return val[0]

	# black_img = np.empty([1200,1600,3],dtype=np.uint8)
	# black_img.fill(0)
	# index = find_contour(contours)
	# cnt = contours[index]
	# mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)
	# plt.imshow(mask)
	# plt.show()
	# maskedImg = cv2.bitwise_and(a, mask)

	# white_pix = [255,255,255]
	# black_pix = [0,0,0]

	# final_img = maskedImg
	# h,w,channels = final_img.shape
	# for x in range(0,w):
	#     for y in range(0,h):
	#         channels_xy = final_img[y,x]
	#         if all(channels_xy == black_pix):    
	#             final_img[y,x] = white_pix


	# plt.imshow(final_img)
	# plt.show()
	# ## Get contour points		
	# contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #2041

	# contour_list = []
	# print len(contours)
	# for contour in contours:
	# 	area = cv2.contourArea(contour)
	# 	if (area >= 100000 and area <= 900000): # 4-9, top 1-9, bot,sides
	# 		contour_list.append(contour)
	# 		lst_intensities = []

	# 		cimg = np.zeros((clone.shape[0],clone.shape[1]), np.uint8)
	# 		cv2.drawContours(cimg, contour_list, -1, color=255, thickness=-1)
	# 			## Cropped Image
	# 		mask_rgb1 = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
	# 		mask = cv2.bitwise_and(clone, mask_rgb1)

	# 			## Replace Garlic to Blue Color
	# 			# indices = np.where(cimg==255)
	# 			# clone[indices[0], indices[1], :] = [0, 0, 255]

	# 			## Append all Image pixels
	# 		indices = np.where(cimg == 255)
	# 		height = indices[0]
	# 		width = indices[1]
	# 		lst_intensities.append(gray[height,width])
	# 		arr = lst_intensities[0]
	# 			# Pixel values of the Garlic Object
	# 		print arr
	# 		# print len(contour_list)
	# 		# if len == 2

	# 			# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/thresholded_bot_{}.png".format(len(images)+1), cimg)
	# 			# cv2.imwrite("C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Segmentation/Results/cropped_bot_{}.png".format(len(images)+1), mask)

	# 		plt.imshow(cimg, cmap='gray')
	# 		plt.show()

	# images.append(arr)

cv2.waitKey(0)
cv2.destroyAllWindows()