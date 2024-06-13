# load image
# print image
# crop
# save

import cv2
img = cv2.imread('student-card.jpg')
img_cropped = img[0:200, 0:200]

cv2.imshow('Before Crop', img)
cv2.imshow('image', img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
