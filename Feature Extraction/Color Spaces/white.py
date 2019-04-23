import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the image
image = cv2.imread('C:/Users/Mynha/Desktop/Dataset/batanes/2018-11-06 07.56.07 1.jpg')

image = cv2.GaussianBlur(image,(5,5),0)

# define the list of boundaries
boundaries = [
	([217, 217, 217], [255, 255, 255])
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# show the images
	plt.imshow(output)
	plt.show()
	cv2.waitKey(0)