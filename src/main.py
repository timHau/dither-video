import cv2

i = cv2.imread('./input/fft2.jpeg', 0)
cv2.imshow('image', i)
cv2.waitKey(0)