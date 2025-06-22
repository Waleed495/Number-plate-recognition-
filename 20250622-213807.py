
import cv2
import numpy as np
import pytesseract

img = cv2.imread('license_plate_image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sobel = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if area > 1000 and aspect_ratio > 2 and aspect_ratio < 6:
        plate_region = gray[y:y+h, x:x+w]
        plate_number = pytesseract.image_to_string(plate_region, config='--psm 11')
        print("License Plate Number:", plate_number.strip())
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('License Plate Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()