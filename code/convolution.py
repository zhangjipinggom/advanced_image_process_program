#ZJP
#convolution.py  19:55
import numpy as np
import cv2

sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
print(sobel_y.shape)
sobel_widths = sobel_y.shape[0]
sobel_heights = sobel_y.shape[1]
print(sobel_y[0][0])

img = cv2.imread("pepper_gray.jpg")
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_widths = grayimg.shape[0]
img_heights = grayimg.shape[1]
#############特别要注意重新赋值##########
new_img = np.zeros([img_widths, img_heights])

# for i in range(1, img_widths-1):
#     for j in range(1, img_heights-1):
#         for m in range(sobel_widths):
#             for n in range(sobel_heights):
#                 new_img[i, j] += grayimg[i+(m-1), j+(n-1)]*sobel_y[m, n]
#
# cv2.imwrite("sobel.jpg", new_img)
new_img2 = cv2.filter2D(grayimg,-1,sobel_y)
cv2.imwrite("sobel2.jpg",new_img2)



