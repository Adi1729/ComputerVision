'''
source : https://www.youtube.com/watch?v=i_30im3FlCs&t=207s
'''

import cv2
import imutils
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = '/home/aditya/Downloadstesseract-ocr-setup-3.02.02.exe'


image = cv2.imread('car10.jpeg')

image = imutils.resize(image , width = 500)

cv2.imshow("Original Image", image)

cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)
cv2.waitKey(0)


gray = cv2.bilateralFilter(gray, 11,17,17)
cv2.imshow("Smoother Image", gray)
cv2.waitKey(0)

# Edges of images

edged = cv2.Canny(gray,170,200)
cv2.imshow("Canny Edge",edged)
cv2.waitKey(0)

# now we will  find the contours based on the images
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()

cv2.drawContours(image1, cnts, -1, (0,255,0), 3)
cv2.imshow("Canny after contouring" ,image1)
cv2.waitKey(0)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0,255,0), 3)
cv2.imshow("Top 30 contours" ,image1)
cv2.waitKey(0)

count = 0
name =1
crop_img_loc = './output/72.png'
img_37 = cv2.imread(crop_img_loc)
cv2.imshow("Cropped Image", img_37)

cv2.waitKey(0)

text  = pytesseract.image_to_string(crop_img_loc,lang='eng')
print(text)
cv2.waitKey(0)

'''
for index,i in enumerate(cnts):
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
#    if(len(approx)==4):  # 4 : Number of corners , Number plate has 4 corners
    NumberPlateCount = approx
    # crop rectangular part
    x,y,w,h = cv2.boundingRect(i)
    crp_img = image[y:y+h,x:x+w]
    print(f'name: {index} perimeter: {perimeter}, corners : {len(approx)}')
    cv2.imwrite('./output/'+str(index)+'.png', crp_img)
    name +=1

cv2.drawContours(image, [cv2.approxPolyDP(cnts[32], 0.02*cv2.arcLength(cnts[32], True), True)], -1, (0,255,0),3)
cv2.imshow("Final Image", image)
cv2.waitKey(0)
'''