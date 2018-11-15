import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
#from binary import Image_to_binary

def nothing(x):
    pass

class Dilate_Erode(object):

    def my_dilate(self,img, kernel):
        kernel_heigh = kernel.shape[0]  # 获取卷积核(滤波)的高度
        kernel_width = kernel.shape[1]  # 获取卷积核(滤波)的宽度
        if kernel_heigh == 0:
            return  img

        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if (img[i, j] < 200):
                    img[i, j] = 0
                else:
                    img[i, j] = 255
        print(img.shape[0], img.shape[1])
        dilate_heigh = img.shape[0] - kernel.shape[0] + 1  # 确定卷积结果的大小
        dilate_width = img.shape[1] - kernel.shape[1] + 1

        dilation = np.zeros((dilate_heigh+kernel_heigh, dilate_width+kernel_width), dtype='uint8')  #'uint8' 'float64'
        #np.set_printoptions(threshold=np.NaN)

        for i in range(dilate_heigh):
            for j in range(dilate_width):   # 如果是白点，就直接覆盖
                if img[i, j] ==  255:
                    dilation[i:i + kernel_heigh, j:j + kernel_width] = kernel
                    #print(dilation)
        dilation = np.delete(dilation,[dilate_heigh,dilate_heigh + kernel_heigh],axis=0)      #去除新加入的列
        dilation = np.delete(dilation,[dilate_width, dilate_width + kernel_width], axis=1)  #去除新加入的行
        print(dilation.shape[0],dilation.shape[1])
        return dilation

    def my_erode(self,img, kernel):

        kernel_heigh = kernel.shape[0]  # 获取卷积核(滤波)的高度
        kernel_width = kernel.shape[1]  # 获取卷积核(滤波)的宽度
        if kernel_heigh == 0:
            return  img

        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if (img[i, j] < 200):
                    img[i, j] = 0
                else:
                    img[i, j] = 255
        print(img.shape[0], img.shape[1])

        erode_heigh = img.shape[0] - kernel.shape[0] + 1  # 确定卷积结果的大小
        erode_width = img.shape[1] - kernel.shape[1] + 1

        erroding = np.zeros((erode_heigh+kernel_heigh, erode_width+kernel_width), dtype='float64')  #'uint8'
        for i in range(erode_heigh):
            for j in range(erode_width):  # 逐点相乘并求和得到每一个点
                erroding[i][j] = self.wise_element_and(img[i:i + kernel_heigh, j:j + kernel_width], kernel)
                #print(i,j)
        erroding = np.delete(erroding,[erode_heigh-1,erode_heigh] , axis=0)  # 去除新加入的列
        erroding = np.delete(erroding, [erode_heigh-1, erode_heigh], axis=0)  # 去除新加入的列
        erroding = np.delete(erroding, [erode_width-1, erode_width], axis=1)  # 去除新加入的列
        erroding = np.delete(erroding, [erode_width-1, erode_width], axis=1)  # 去除新加入的列
        #erroding = np.delete(erroding, [0,3] , axis=1)  # 去除新加入的行
        print(erroding.shape[0], erroding.shape[1])
        return erroding

    def wise_element_and(self,img, kernel):

        res = np.logical_and(img,kernel)
        boolv = res.all()
        if boolv == 1:
            pixel = 255
        else:
            pixel = 0
        return pixel

    def Opening(self,img,kernel):
        erroding = self.my_erode(img, kernel)
        opening = self.my_dilate(erroding, kernel)
        return opening

    def Closing(self,img,kernel):
        dilation = self.my_dilate(img, kernel)
        closing = self.my_erode(dilation, kernel)
        return closing

#while (1):
#s = cv2.getTrackbarPos('Er/Di', 'image')
#si = cv2.getTrackbarPos('size', 'image')
# 分别接收两个滚动条的数据

'''
while (1):
    cv2.waitKey(0)
    img_init = cv2.imread("D:\MyHomework\ImageProcess\image//chuizi.jpg", 0)
    cv2.imshow("init",img_init)
    de = Dilate_Erode()
    imgtobi = Image_to_binary()
    kernel = np.ones((3, 3), np.uint8)
    kernel = kernel*255
    histogram, img = imgtobi.img_to_bin(img_init, 200)

    #erroding = de.Closing(img, kernel)
    #erroding = de.Opening(erroding, kernel)
    #erroding = de.my_dilate(img, kernel)
    erroding = de.my_dilate(img, kernel)

    cv2.imshow("errod", erroding)
'''

