#

import cv2 as cv
import numpy as np
import imutils
import pytesseract
#import easyocr



img = cv.imread('images/cropped_parking_lot_10.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Converte a imagem para tons de cinza
img_blur = cv.GaussianBlur(gray, (5, 5), 0)

# edged = img_blur.astype(np.float32) - gray.astype(np.float32)
# edged = np.where(edged < 2, 0, edged)
# edged = np.absolute(edged).astype(np.uint8)
# edged = cv.equalizeHist(edged)
# _, edged = cv.threshold(edged, 127, 255, cv.THRESH_BINARY)

edged = cv.Canny(img_blur, 70, 150) #Edge detection
struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)

# Dilata a imagem
edged = cv.dilate(edged, struct_elem, iterations = 1)

#cv.imshow('IMG_EDGES', edged)

# encontra os contornos da imagem
keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

img_copy = img.copy()
location = None
i = 0
# Para cada contorno encontrado na imagem
for contour in contours:
  approx = cv.approxPolyDP(contour, 10, True)
  if len(approx) == 4:
    location = approx
  
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0,255, -1)
    new_image = cv.bitwise_and(img, img, mask=mask)

    # i = i + 1
    # imgName = 'NEW_IMG' + str(i)
    # cv.imshow(imgName, new_image)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    i = i + 1
    imgName = 'IMG_CROP' + str(i)
    cv.imshow(imgName, cropped_image)

    # Faz a limiarização da imagem para convertê-la em uma imagem binária
    _, cropped_image = cv.threshold(cropped_image, 50, 255, cv.THRESH_BINARY)

    struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    cropped_image = cv.erode(cropped_image, struct_elem, iterations = 3)
    # cropped_image = cv.morphologyEx(cropped_image, cv.MORPH_CLOSE, struct_elem)
    # cropped_image = cv.morphologyEx(cropped_image, cv.MORPH_CLOSE, struct_elem)
    # cropped_image = cv.morphologyEx(cropped_image, cv.MORPH_CLOSE, struct_elem)

    imgName = 'IMG_BIN' + str(i)
    cv.imshow(imgName, cropped_image)

    # Configura o Pytesseract
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
    
    result = pytesseract.image_to_string(cropped_image, lang='eng', config=config)
    result = result.replace("\n","").replace("\f","")

    if len(result) >= 2:
      text = result
      font = cv.FONT_HERSHEY_SIMPLEX
      res = cv.putText(img_copy, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
      res = cv.rectangle(img_copy, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    
cv.imshow('Result', img_copy)

if location is None:
  detected = 0
  print ("No contour detected")

cv.waitKey(0)
cv.destroyAllWindows()