#

import cv2 as cv
import numpy as np
import imutils
import pytesseract
#import easyocr

img = cv.imread('carro5.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('IMG', img)
cv.imshow('IMG_GRAY', gray)

img_blur = cv.GaussianBlur(gray, (5, 5), 0)
# bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv.Canny(img_blur, 100, 200) #Edge detection

cv.imshow('IMG_EDGES', edged)

# encontra os contornos da imagem
keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in contours:
  approx = cv.approxPolyDP(contour, 10, True)
  if len(approx) == 4:
    location = approx
    break

cv.waitKey(0)

if location is None:
  detected = 0
  print ("No contour detected")

mask = np.zeros(gray.shape, np.uint8)
new_image = cv.drawContours(mask, [location], 0,255, -1)
new_image = cv.bitwise_and(img, img, mask=mask)

cv.imshow('NEW_IMG', new_image)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

cv.imshow('CROPPED_IMG', cropped_image)

config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
result = pytesseract.image_to_string(cropped_image, lang='eng', config=config)

text = result
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
cv.imshow('Result', res)


cv.waitKey(0)
cv.destroyAllWindows()