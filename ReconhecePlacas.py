#

import cv2 as cv
import numpy as np
import imutils
import pytesseract
import os

debug = True

def obtem_lista_imagens():
  return os.listdir('images/')

def detecta_reconhece_placa(img_file_name):
  img = cv.imread(img_file_name)

  # Converte a imagem para tons de cinza
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  img_blur = cv.GaussianBlur(gray, (5, 5), 0)

  edged = cv.Canny(img_blur, 70, 150) #Edge detection

  # Dilata a imagem
  struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
  edged = cv.dilate(edged, struct_elem, iterations = 1)

  if (debug):
    cv.imshow('IMG_EDGES', edged)

  # encontra os contornos da imagem
  keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

  img_copy = img.copy()
  location = None
  i = 0
  # Para cada contorno encontrado na imagem
  text_result = ""
  for contour in contours:
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.018 * peri, True)
    if len(approx) == 4:
      location = approx
    
      mask = np.zeros(gray.shape, np.uint8)
      new_image = cv.drawContours(mask, [location], 0,255, -1)
      new_image = cv.bitwise_and(img, img, mask=mask)

      i = i + 1
      if(debug):
        imgName = 'NEW_IMG' + str(i)
        cv.imshow(imgName, new_image)

      (x,y) = np.where(mask==255)
      (x1, y1) = (np.min(x), np.min(y))
      (x2, y2) = (np.max(x), np.max(y))
      cropped_image = gray[x1:x2+1, y1:y2+1]

      if(debug):
        imgName = 'IMG_CROP' + str(i)
        cv.imshow(imgName, cropped_image)

      # escala a imagem
      cropped_image = cv.resize(cropped_image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

      # Faz a limiarização da imagem para convertê-la em uma imagem binária
      _, cropped_image = cv.threshold(cropped_image, 50, 255, cv.THRESH_BINARY)

      # erosão para aprimorar as letras (pretas)
      struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
      cropped_image = cv.erode(cropped_image, struct_elem, iterations = 3)

      if(debug):
        imgName = 'IMG_BIN' + str(i)
        cv.imshow(imgName, cropped_image)

      # Configura o Pytesseract
      config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
      
      result = pytesseract.image_to_string(cropped_image, lang='eng', config=config)
      result = result.replace("\n","").replace("\f","")

      if len(result) >= 2:
        text_result = result
        text = result
        font = cv.FONT_HERSHEY_SIMPLEX
        res = cv.putText(img_copy, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        res = cv.rectangle(img_copy, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
      
  cv.imshow('Result', img_copy)

  cv.waitKey(0)
  cv.destroyAllWindows()

  if location is None:
    detected = 0
    print ("No contour detected")

  return text_result

if __name__ == "__main__":
  lista_imagens = obtem_lista_imagens()
  lista_result = []
  for image_file_name in lista_imagens:
    print(image_file_name)
    lista_result.append(detecta_reconhece_placa("images/" + image_file_name))
  print(lista_result)
