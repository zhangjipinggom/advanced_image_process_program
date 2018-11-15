#ZJP
#project1.py  10:32
import cv2

class PROJECT1:
    def __init__(self):
        pass

    def showHistogram(self, source_image):
        import numpy as np
        import matplotlib.pyplot as plt
        img_rows = source_image.shape[0]
        img_columns = source_image.shape[1]
        total_pixel_num = img_rows * img_columns
        pixel_num = np.zeros(256)
        gray_level_show = np.zeros(256)
        for gray_inedex in range(256):
            gray_level_show[gray_inedex] = gray_inedex
        for i in range(self.img_width):
            for j in range(self.img_height):
                for gray_level in range(256):
                    if self.grayimg_histogram[i][j] == gray_level:
                        self.pixel_num[gray_level] = self.pixel_num[gray_level] + 1
        plt.figure()
        plt.bar(x=gray_level_show, height=self.pixel_num, color='b', align="center", yerr=0.001)
        plt.xlabel("Gray level")
        plt.ylabel("Pixel Number")
        plt.savefig("histogram.jpg")





