import cv2
img = cv2.imread('1.jpeg')

cv2.imshow('image', img)
imgresize = cv2.resize(img, (566, 800))    
cv2.imshow('image', imgresize)

cv2.waitKey(0)
cv2.destroyAllWindows()