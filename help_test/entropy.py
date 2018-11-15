#ZJP
#entropy.py  22:08


#ZJP
#OTSU.py  18:52

import cv2
import numpy as np
import math
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

possibility1 = np.zeros(256)
possibility2 = np.zeros(256)
log_p1 = np.zeros(256)
log_p2 = np.zeros(256)
total_num1 = np.zeros(256)
total_num2 = np.zeros(256)
H1 = np.zeros(256)
H2 = np.zeros(256)
H = np.zeros(256)


for entropy_threshold in range(256):
    total_num1[entropy_threshold] += pixel_num[entropy_threshold]
    total_num2[entropy_threshold] = total_pixel_num - total_num1[entropy_threshold]
    if total_num1[entropy_threshold] == 0:
        possibility1[entropy_threshold] = 0
    if total_num1[entropy_threshold] >= 0:
        for entropy_threshold_change in range(0, entropy_threshold):        # 实现累加
            possibility1[entropy_threshold] = pixel_num[entropy_threshold_change]/total_num1[entropy_threshold]
            log_p1[entropy_threshold] = np.log(possibility1[entropy_threshold])    # 防止对负数取对数
            H1[entropy_threshold] += -possibility1[entropy_threshold]*log_p1[entropy_threshold]
    if total_num2[entropy_threshold] == 0:
        possibility2[entropy_threshold] = 0
    if total_num2[entropy_threshold] >= 0:
        for entropy_threshold_change2 in range(entropy_threshold, 256):  # 实现累加
            possibility2[entropy_threshold] = pixel_num[entropy_threshold_change2]/total_num2[entropy_threshold]
            log_p2[entropy_threshold] = np.log(possibility2[entropy_threshold])
            H2[entropy_threshold] += -possibility2[entropy_threshold]*log_p2[entropy_threshold]
    H[entropy_threshold] = H1[entropy_threshold] + H2[entropy_threshold]

print(max(H))
a = max(H)
for entropy_threshold2 in range(256):
    if H[entropy_threshold2] == a:
        print(entropy_threshold2)
        entropy_threshold_true = entropy_threshold2

for m in range(img_width):
    for n in range(img_height):
        if grayimg[m][n]>=entropy_threshold_true:
            grayimg[m][n] = 255
        else:
            grayimg[m][n] = 0
#cv2.imshow("Entropy", grayimg)
cv2.imwrite("Entropy.jpg", grayimg)



