#ZJP
#OTSU.py  18:52

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "camera_gray.jpg"
img = cv2.imread(image_path)
#img = cv2.resize(img, (30, 30))
#grayimg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(grayimg)
# plt.show()
img_width = grayimg.shape[0]
img_height = grayimg.shape[1]
pixel_num = np.zeros(256)
gray_level_show = np.zeros(256)
for gray_inedex in range(256):
    gray_level_show[gray_inedex]=gray_inedex
print(gray_level_show)
for i in range(img_width):
    for j in range(img_height):
        for gray_level in range(256):
            if grayimg[i][j] == gray_level:
                pixel_num[gray_level] = pixel_num[gray_level]+1

print(pixel_num)
total_pixel_num = img_height*img_width

possibility = pixel_num / total_pixel_num
variance = np.zeros(256)
frequency1 = np.zeros(256)
frequency2 = np.zeros(256)
mean1 = np.zeros(256)
mean2 = np.zeros(256)
class_difference = np.zeros(256)
for otsu_threshold in range(256):
     frequency1[otsu_threshold] = frequency1[otsu_threshold]+possibility[otsu_threshold]
     frequency2[otsu_threshold] = 1-frequency1[otsu_threshold]
     if frequency1[otsu_threshold] == 0:
         mean1[otsu_threshold] = 0
     if frequency1[otsu_threshold] > 0:
        mean1[otsu_threshold] += otsu_threshold*possibility[otsu_threshold]/frequency1[otsu_threshold]
     for background in range(otsu_threshold, 256):
         mean2[otsu_threshold] += background*possibility[background]
     if frequency2[otsu_threshold] == 0:
         mean2 = 0
     if frequency2[otsu_threshold] > 0:
         mean2[otsu_threshold] = mean2[otsu_threshold]/frequency2[otsu_threshold]
     class_difference[otsu_threshold] = mean1[otsu_threshold]-mean2[otsu_threshold]
     variance[otsu_threshold] = \
         frequency1[otsu_threshold]*frequency2[otsu_threshold]*class_difference[otsu_threshold]*class_difference[otsu_threshold]

#otsu_threshold_true = max(variance)
print(max(variance))
a = max(variance)
for otsu_threshold2 in range(256):
    if variance[otsu_threshold2] == a:
        print(otsu_threshold2)
        otsu_threshold_true =otsu_threshold2


for m in range(img_width):
    for n in range(img_height):
        if grayimg[m][n]>=otsu_threshold_true:
            grayimg[m][n] = 255
        else:
            grayimg[m][n] = 0
print(grayimg)
cv2.imshow("OTSU", grayimg)
cv2.imwrite("OTSU.jpg",grayimg)



