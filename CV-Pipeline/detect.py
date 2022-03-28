import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local
from imutils.perspective import four_point_transform

image = cv2.imread('test1.png')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

kernel = np.ones((5,5), np.uint8)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ero = cv2.dilate(im_bw, kernel, iterations=3)
# h, w = ero.shape
# ero = cv2.rectangle(ero,(0,h-15),(w,h),(0,0,0),15)
cv2.imshow("aa", ero)

# x, y, w, h = cv2.boundingRect(ero)
# left = (x, np.argmax(ero[:, x]))
# right = (x+w-1, np.argmax(ero[:, x+w-1]))
# top = (np.argmax(ero[y, :]), y)
# bottom = (np.argmax(ero[y+h-1, :]), y+h-1)

# points = np.array([left, right, bottom, top])
# print(points)
# warped = four_point_transform(orig, points)


cnts = cv2.findContours(ero.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break
print(screenCnt.reshape(4, 2) * ratio)

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
