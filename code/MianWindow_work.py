#ZJP
#MianWindow_work.py  19:13

from PyQt5.QtGui import QPixmap, QFont, QPalette, QImage, QPainter
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QInputDialog, QSplashScreen, QFileDialog, QGraphicsScene, \
    QGraphicsPixmapItem
from PyQt5.QtCore import pyqtSlot, Qt, QFile, QSize
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from MianWindow_zjp2 import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("icon_mainwindow.png"))
        QtWidgets.QToolTip.setFont(QFont('Arial',10))
        self.graphicsView_1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


        self.scene = QGraphicsScene()
        self.size_flag_1 = 0
        self.size_flag_2 = 0
        self.cross_flag = 0
        self.ellipse_flag = 0
        self.origin_flag = 0
        self.binary_flag = 0
        # self.edge_st = 0
        self.edge_ex_flag = 0
        self.edge_in_flag = 0
        self.gradient_ex_flag = 0
        self.gradient_in_flag = 0
        self.obr_flag = 0


        ################设置提示功能############
        self.pushButton_histogram.setToolTip("\nHistogram Button\n"
                                             "if you press the button,\n"
                                             "the histogram of the image will be analysed and displayed\n")
        self.pushButton_otsu.setToolTip("\nOTSU Button\n"
                                             "if you press the button,\n"
                                             "the image will be classified as \n"
                                        "foreground and background using OTSU method \n")
        self.pushButton_entropy.setToolTip("\nEntropy Method Button\n"
                                             "if you press the button,\n"
                                             "the image will be classified as \n"
                                        "foreground and background using entropy method\n ")

    # def my_clicked(self,event):
    #     print('haha')
    #     webbrowser.open("www.baidu.com")
    #
    # #@pyqtSlot()
    # @pyqtSlot()
    # def on_pushButton_clicked(self):
    #     text_string =  self.textEdit.toPlainText()
    #     #self.textBrowser.setText(text_string)
    #     brower_text = self.textBrowser.toPlainText()
    #     self.textBrowser.append(text_string)
    #     print(brower_text)
    #
    # def on_pushButton2_clicked(self):
    #     #self.textEdit.setText('')
    #     self.textBrowser.property()
    #     self.textBrowser.clear('')
    #
    # def on_pushButton3_clicked(self):
    #     #my_button = QMessageBox.information(self,'hhah','提示')
    #     my_str,ok = QInputDialog.getText(self)
    #

#####################几个控件的信号发射#############################
        self.horizontalScrollBar_scale.valueChanged[int].connect(self.changeValue)          # 图的缩放
        self.pushButton_histogram.clicked.connect(self.showHistogram)          #直方图
        self.pushButton_otsu.clicked.connect(self.showOTSU)               # 大津算法
        self.pushButton_entropy.clicked.connect(self.show_entropy)        # 熵算法
        self.horizontalScrollBar.sliderMoved[int].connect(self.manual_threshold)    # 手动选取阈值二值化
        self.pushButton_sobel_x.clicked.connect(self.show_sobel_x_edge)         # sobel求水平边缘
        self.pushButton_sobel_y.clicked.connect(self.show_sobel_y_edge)         # sobel求垂直边缘
        self.pushButton_previtt_x.clicked.connect(self.show_previtt_x_edge)  # previtt求水平边缘
        self.pushButton_previtt_y.clicked.connect(self.show_previtt_y_edge)  # previtt求垂直边缘
        self.pushButton_meam.clicked.connect(self.show_mean_smooth)  # previtt求垂直边缘
        self.spinBox.valueChanged[int].connect(self.gaussian_size)
        self.doubleSpinBox.valueChanged[float].connect(self.gaussian_sigma)
        self.pushButton_gaussian.clicked.connect(self.show_gaussian_smooth)
        self.pushButton_custom.clicked.connect(self.show_custom_dialog)
        self.pushButton_custom_2.clicked.connect(self.show_custom_3_3)
        self.pushButton_custom_3.clicked.connect(self.show_custom_5_5)


        #### label 2 的按钮 ######################
        self.radioButton_cross.toggled.connect(self.setshape_cross)
        self.radioButton_ellipse.toggled.connect(self.setshape_ellipse)
        self.radioButton_rect.toggled.connect(self.setshape_rect)
        self.horizontalScrollBar_2.sliderMoved[int].connect(self.manual_threshold)


        self.pushButton_erosion.clicked.connect(self.my_erosion)
        self.pushButton_dilation.clicked.connect(self.my_dilation)
        self.pushButton_opening.clicked.connect(self.my_opening)
        self.pushButton_closing.clicked.connect(self.my_closing)
        self.pushButton_distance_transform.clicked.connect(self.distance_transform)
        self.pushButton_skeleton.clicked.connect(self.skeleton)
        self.pushButton_skeleton_restoration.clicked.connect(self.skeleton_restoration)


        ################### toolbox in label3 #####################
        self.radioButton_cross_gray.toggled.connect(self.setshape_cross)
        self.radioButton_ellipse_gray.toggled.connect(self.setshape_ellipse)
        self.radioButton_rect_gray.toggled.connect(self.setshape_rect)
        self.pushButton_erosion_gray.clicked.connect(self.erosion_gray)
        self.pushButton_dilation_gray.clicked.connect(self.dilation_gray)
        self.pushButton_opening_gray.clicked.connect(self.opening_gray)
        self.pushButton_closing_gray.clicked.connect(self.closing_gray)

        ############### toolbox in project6 ################
        self.radioButton_edge_st.toggled.connect(self.set_edge_type_st)
        self.radioButton_edge_ex.toggled.connect(self.set_edge_type_ex)
        self.radioButton_edge_in.toggled.connect(self.set_edge_type_in)
        self.radioButton_gradient_st.toggled.connect(self.set_gradient_type_st)
        self.radioButton_gradient_ex.toggled.connect(self.set_gradient_type_ex)
        self.radioButton_gradient_in.toggled.connect(self.set_gradient_type_in)
        self.pushButton_mor_edge.clicked.connect(self.mor_egde_gray)
        self.pushButton_mor_gradient.clicked.connect(self.mor_gradient_gray)
        self.pushButton_reconstruction.clicked.connect(self.cbr)
        self.horizontalScrollBar_threshold_gray.sliderMoved[int].connect(self.manual_threshold)


    def imshow_ori(self, img_to_show, viewer_index):
        height, width, bytesPerComponent = img_to_show.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB, img_to_show)
        QImg = QImage(img_to_show.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(QImg)
        if viewer_index == 1:
            self.item_1 = QGraphicsPixmapItem(self.pixmap)
            self.scene_1 = QGraphicsScene(self)
            self.scene_1.addItem(self.item_1)
            self.graphicsView_1.setScene(self.scene_1)
        if viewer_index == 2:
            self.scene_2 = QGraphicsScene(self)
            self.item_2 = QGraphicsPixmapItem(self.pixmap)
            self.scene_2.addItem(self.item_2)
            self.graphicsView_2.setScene(self.scene_2)
        if viewer_index == 3:
            self.scene_3 = QGraphicsScene(self)
            self.item_3 = QGraphicsPixmapItem(self.pixmap)
            self.scene_3.addItem(self.item_3)
            self.graphicsView_3.setScene(self.scene_3)
        if viewer_index == 4:
            self.scene_4 = QGraphicsScene(self)
            self.item_4 = QGraphicsPixmapItem(self.pixmap)
            self.scene_4.addItem(self.item_4)
            self.graphicsView_4.setScene(self.scene_4)

    ###############打开菜单的响应函数#################
    @pyqtSlot()
    def on_actionOpen_triggered(self):
        self.my_file_path = QFileDialog.getOpenFileName(self, "open file", 'E:/advanced_image_processing/test_image/')
        filename = self.my_file_path[0]
        print(filename)
        image_class = [".jpg", ".JPG", '.PNG', '.png', '.tif']
        if filename == '':
            QMessageBox.information(self, 'information','You have not open any image')
            pass
        if (os.path.splitext(filename))[1] not in image_class:
            QMessageBox.information(self, 'warning', 'unsupported file format')
            pass
        if os.path.splitext(filename[1]) not in image_class:   # not?
           img = cv2.imread(filename)
           self.grayimg = cv2.imread(filename, flags=0)
           self.imshow_ori(img, 1)


    ###########关闭菜单的响应函数##########
    @pyqtSlot()
    def on_actionQuit_triggered(self):
        sys.exit(0)

    ###########关于我菜单的响应函数##########
    @pyqtSlot()
    def on_actionAbout_Me_triggered(self):
        QMessageBox.about(self, 'Introduction', 'This is a small program that \n'
                                                'achieve some  simple image processing functions')

    ##########联系我菜单的响应函数##########
    @pyqtSlot()
    def on_actionContact_Me_triggered(self):
        QMessageBox.about(self, 'Communication', 'any question? \n'
                                                 'welcome send email to zhangjipinggom@163.com')

    #################拖动滑块的响应函数：实现图片的缩小放大功能##################
    def changeValue(self, value):
        scale = value*0.1

        a = self.pixmap.size()
        Item_width = a.width()*scale
        Item_heigh = a.height()*scale
        self.item_2.setScale(scale)
        self.item_3.setScale(scale)
        self.item_4.setScale(scale)
        self.item_1.setScale(scale)
        self.scene.setSceneRect(0, 0, Item_width, Item_heigh)
        # self.scene_2.setSceneRect(0, 0, Item_width, Item_heigh)
        # self.scene_3.setSceneRect(0, 0, Item_width, Item_heigh)
        # self.scene_4.setSceneRect(0, 0, Item_width, Item_heigh)

    ################画直方图按钮的响应函数##############
    def showHistogram(self):
        image_path = self.my_file_path[0]
        img = cv2.imread(image_path)
        self.grayimg_histogram = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_width = self.grayimg_histogram.shape[0]
        self.img_height = self.grayimg_histogram.shape[1]
        self.total_pixel_num = self.img_width*self.img_height
        self.pixel_num = np.zeros(256)
        gray_level_show = np.zeros(256)
        pixel_level = 256
        for gray_inedex in range(256):
            gray_level_show[gray_inedex] = gray_inedex
        for i in range(self.img_width):
            for j in range(self.img_height):
                for gray_level in range(0, pixel_level):
                    if self.grayimg_histogram[i][j] == gray_level:
                        self.pixel_num[gray_level] = self.pixel_num[gray_level] + 1
        plt.figure()
        plt.bar(x=gray_level_show, height=self.pixel_num, color='b', align="center", yerr=0.001)
        # plt.hist(grayimg.ravel(), 255)       # 快
        plt.xlabel("Gray level")
        plt.ylabel("Pixel Number")
        plt.savefig("histogram.jpg")
        histogram_large = cv2.imread("histogram.jpg")
        histogram_large = cv2.resize(histogram_large, (350, 280))
        height, width, bytesPerComponent =histogram_large.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(histogram_large, cv2.COLOR_BGR2RGB, histogram_large)
        QImg = QImage(histogram_large.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_histogram = QPixmap.fromImage(QImg)
        self.scene_5 = QGraphicsScene(self)
        self.item_histogram = QGraphicsPixmapItem(self.pixmap_histogram)
        self.scene_5.addItem(self.item_histogram)
        self.scene_5.addText("\nHistogram", QFont('Arial',8))
        self.graphicsView_histogram.setScene(self.scene_5)

    ##################大津算法二值化###############
    def showOTSU(self):
        image_path = self.my_file_path[0]
        self.grayimg_otsu = cv2.imread(image_path, flags=0)
        # self.grayimg_otsu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        possibility = self.pixel_num / self.total_pixel_num
        variance = np.zeros(256)
        frequency1 = np.zeros(256)
        frequency2 = np.zeros(256)
        mean1 = np.zeros(256)
        mean2 = np.zeros(256)
        class_difference = np.zeros(256)
        #######计算方差##########
        for otsu_threshold in range(256):
            frequency1[otsu_threshold] = frequency1[otsu_threshold] + possibility[otsu_threshold]
            frequency2[otsu_threshold] = 1 - frequency1[otsu_threshold]
            if frequency1[otsu_threshold] == 0:
                mean1[otsu_threshold] = 0
            if frequency1[otsu_threshold] > 0:
                mean1[otsu_threshold] += otsu_threshold * possibility[otsu_threshold] / frequency1[otsu_threshold]
            for background in range(otsu_threshold, 256):
                mean2[otsu_threshold] += background * possibility[background]
            if frequency2[otsu_threshold] == 0:
                mean2 = 0
            if frequency2[otsu_threshold] > 0:
                mean2[otsu_threshold] = mean2[otsu_threshold] / frequency2[otsu_threshold]
            class_difference[otsu_threshold] = mean1[otsu_threshold] - mean2[otsu_threshold]
            variance[otsu_threshold] = \
                frequency1[otsu_threshold] * frequency2[otsu_threshold] * class_difference[otsu_threshold] * \
                class_difference[otsu_threshold]
        print(max(variance))
        variance_max = max(variance)
        ####找出阈值####
        for otsu_threshold2 in range(256):
            if variance[otsu_threshold2] == variance_max:
                print(otsu_threshold2)
                otsu_threshold_true = otsu_threshold2
         ######二值化#########
        for m in range(self.img_width):
            for n in range(self.img_height):
                if self.grayimg_otsu[m][n] >= otsu_threshold_true:
                    self.grayimg_otsu[m][n] = 255
                else:
                    self.grayimg_otsu[m][n] = 0
        cv2.imwrite("OTSU.jpg", self.grayimg_otsu)
        OTSU_img = cv2.imread("OTSU.jpg")
        self.imshow_ori(OTSU_img, viewer_index=3)
        otsu_str = str(otsu_threshold_true)
        self.textBrowser.append("Threshold in OTSU method-"+otsu_str)
        self.textBrowser.append("\n")

    ##################熵的方法二值化##################
    def show_entropy(self):
        image_path = self.my_file_path[0]
        img = cv2.imread(image_path)
        self.grayimg_entropy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        possibility1 = np.zeros(256)
        possibility2 = np.zeros(256)
        log_p1 = np.zeros(256)
        log_p2 = np.zeros(256)
        total_num1 = np.zeros(256)
        total_num2 = np.zeros(256)
        H1 = np.zeros(256)
        H2 = np.zeros(256)
        H = np.zeros(256)
        #######计算方差##########
        for entropy_threshold in range(256):
            total_num1[entropy_threshold] += self.pixel_num[entropy_threshold]
            total_num2[entropy_threshold] = self.total_pixel_num - total_num1[entropy_threshold]
            if total_num1[entropy_threshold] == 0:
                possibility1[entropy_threshold] = 0
            if total_num1[entropy_threshold] >= 0:
                for entropy_threshold_change in range(0, entropy_threshold):  # 实现累加
                    possibility1[entropy_threshold] = self.pixel_num[entropy_threshold_change] / \
                                                      (total_num1[entropy_threshold]+1e-6)
                    log_p1[entropy_threshold] = np.log(possibility1[entropy_threshold]+1e-6)  # 防止对负数取对数
                    H1[entropy_threshold] += -possibility1[entropy_threshold] * log_p1[entropy_threshold]
            if total_num2[entropy_threshold] == 0:
                possibility2[entropy_threshold] = 0
            if total_num2[entropy_threshold] >= 0:
                for entropy_threshold_change2 in range(entropy_threshold, 256):  # 实现累加
                    possibility2[entropy_threshold] = self.pixel_num[entropy_threshold_change2] /\
                                                      (total_num2[entropy_threshold]+1e-6)
                    log_p2[entropy_threshold] = np.log(possibility2[entropy_threshold]+1e-6)
                    H2[entropy_threshold] += -possibility2[entropy_threshold] * log_p2[entropy_threshold]
            H[entropy_threshold] = H1[entropy_threshold] + H2[entropy_threshold]
        print(max(H))
        a = max(H)
        for entropy_threshold2 in range(256):
            if H[entropy_threshold2] == a:
                print(entropy_threshold2)
                entropy_threshold_true = entropy_threshold2

        for m in range(self.img_width):
            for n in range(self.img_height):
                if self.grayimg_entropy[m][n] >= entropy_threshold_true:
                    self.grayimg_entropy[m][n] = 255
                else:
                    self.grayimg_entropy[m][n] = 0
        cv2.imwrite("entropy.jpg", self.grayimg_entropy)
        entropy_img = cv2.imread("entropy.jpg")
        self.imshow_ori(entropy_img, viewer_index=4)
        entropy_str = str(entropy_threshold_true)
        self.textBrowser.append("Threshold in entropy method-"+entropy_str)
        self.textBrowser.append("\n")

    ################## 滑块选取阈值进行手动分割######################
    def manual_threshold(self, value):
        image_path = self.my_file_path[0]
        img = cv2.imread(image_path, flags=0)
        self.binary_flag = 1
        self.grayimg_manual = img
        manual_threshold_true = value
        rows, columns = img.shape
        for m in range(rows):
            for n in range(columns):
                if self.grayimg_manual[m][n] >= manual_threshold_true:
                    self.grayimg_manual[m][n] = 255
                else:
                    self.grayimg_manual[m][n] = 0
        cv2.imwrite("manual.jpg", self.grayimg_manual)
        self.manual_img = cv2.imread("manual.jpg")
        self.imshow_ori(self.manual_img, viewer_index=2)
        manual_str = str(manual_threshold_true)
        self.textBrowser.append("Threshold selected-"+manual_str)
        self.textBrowser.append("\n")

    #################### sobel x edge ########################
    def show_sobel_x_edge(self):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_widths = sobel_x.shape[0]
        sobel_heights = sobel_x.shape[1]
        img_widths = self.grayimg.shape[0]
        img_heights = self.grayimg.shape[1]
        #############特别要注意重新赋值##########
        new_sobel_x_img = np.zeros([img_widths, img_heights])
        for i in range(1, img_widths-1):
            for j in range(1, img_heights-1):
                for m in range(sobel_widths):
                    for n in range(sobel_heights):
                        new_sobel_x_img[i, j] += self.grayimg[i+(m-1), j+(n-1)]*sobel_x[m, n]
        cv2.imwrite("sobel_x.jpg", new_sobel_x_img)
        sobel_x_img = cv2.imread("sobel_x.jpg")
        self.imshow_ori(sobel_x_img, viewer_index=2)

    ############### sobel y edge ##################
    def show_sobel_y_edge(self):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T
        new_sobel_y_img = cv2.filter2D(self.grayimg, -1, sobel_y)  ######## 自带的，应该是做过了优化，速度快
        cv2.imwrite("sobel_y.jpg", new_sobel_y_img)
        sobel_y_img = cv2.imread("sobel_y.jpg")
        self.imshow_ori(sobel_y_img, 2)

    ############### previtt y edge ################
    def show_previtt_y_edge(self):
        previtt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        previtt_y = previtt_x.T
        new_previtt_y_img = cv2.filter2D(self.grayimg, -1, previtt_y)  ######## 自带的，应该是做过了优化，速度快
        cv2.imwrite("previtt_y.jpg", new_previtt_y_img)
        previtt_y_img = cv2.imread("previtt_y.jpg")
        height, width, bytesPerComponent = previtt_y_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(previtt_y_img, cv2.COLOR_BGR2RGB, previtt_y_img)
        QImg = QImage(previtt_y_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_previtt_y = QPixmap.fromImage(QImg)
        self.scene_2 = QGraphicsScene(self)
        self.item_2 = QGraphicsPixmapItem(self.pixmap_previtt_y)
        self.scene_2.addItem(self.item_2)
        self.graphicsView_2.setScene(self.scene_2)

    ############### previtt x edge ################
    def show_previtt_x_edge(self):
        previtt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        new_previtt_y_img = cv2.filter2D(self.grayimg, -1, previtt_x)  ######## 自带的，应该是做过了优化，速度快
        cv2.imwrite("previtt_x.jpg", new_previtt_y_img)
        previtt_x_img = cv2.imread("previtt_x.jpg")
        height, width, bytesPerComponent = previtt_x_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(previtt_x_img, cv2.COLOR_BGR2RGB, previtt_x_img)
        QImg = QImage(previtt_x_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_previtt_y = QPixmap.fromImage(QImg)
        self.scene_2 = QGraphicsScene(self)
        self.item_2 = QGraphicsPixmapItem(self.pixmap_previtt_y)
        self.scene_2.addItem(self.item_2)
        self.graphicsView_2.setScene(self.scene_2)

    ########  均值滤波 #########
    def show_mean_smooth(self):
        mean_filter = np.ones([3, 3])
        mean_filter = mean_filter/9
        mean_img = cv2.filter2D(self.grayimg, -1, mean_filter)  ######## 自带的，应该是做过了优化，速度快
        cv2.imwrite("mean.jpg", mean_img)
        mean_img = cv2.imread("mean.jpg")
        height, width, bytesPerComponent = mean_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(mean_img, cv2.COLOR_BGR2RGB, mean_img)
        QImg = QImage(mean_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_mean = QPixmap.fromImage(QImg)
        self.scene_3 = QGraphicsScene(self)
        self.item_3 = QGraphicsPixmapItem(self.pixmap_mean)
        self.scene_3.addItem(self.item_3)
        self.graphicsView_3.setScene(self.scene_3)

    ####### 高斯核大小########
    def gaussian_size(self, value):
        if value % 2 == 0:
            QMessageBox.warning("please input odd number")
        if value % 2 == 1:
            self.g_filter_size = value
        g_filter_sigma = 1.00
        g_filter_size = self.g_filter_size
        gaussian_kernal = np.zeros([g_filter_size, g_filter_size])
        for i in range(g_filter_size):
            for j in range(g_filter_size):
                math_i_index = i - int((g_filter_size + 1) / 2)
                math_j_index = j - int((g_filter_size + 1) / 2)
                gaussian_kernal[i, j] = (1 / (2 * np.pi * g_filter_sigma ** 2)) * \
                                        math.exp(-(math_i_index ** 2 + math_j_index ** 2) / (2 * (g_filter_sigma ** 2)))
        new_gaussian_img = cv2.filter2D(self.grayimg, -1, gaussian_kernal)
        cv2.imwrite("gaussian.jpg", new_gaussian_img)
        gaussian_img = cv2.imread("gaussian.jpg")
        height, width, bytesPerComponent = gaussian_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB, gaussian_img)
        QImg = QImage(gaussian_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_gaussian = QPixmap.fromImage(QImg)
        self.scene_3 = QGraphicsScene(self)
        self.item_3 = QGraphicsPixmapItem(self.pixmap_gaussian)
        self.scene_3.addItem(self.item_3)
        self.graphicsView_3.setScene(self.scene_3)

    ############ 高斯方差大小 依次变化###############
    def gaussian_sigma(self, value):
        self.g_filter_sigma = value
        g_filter_sigma = self.g_filter_sigma
        g_filter_size = 3
        gaussian_kernal = np.zeros([g_filter_size, g_filter_size])
        for i in range(g_filter_size):
            for j in range(g_filter_size):
                math_i_index = i - int((g_filter_size + 1) / 2)
                math_j_index = j - int((g_filter_size + 1) / 2)
                gaussian_kernal[i, j] = (1 / (2 * np.pi * g_filter_sigma ** 2)) * \
                                        math.exp(-(math_i_index ** 2 + math_j_index ** 2) / (2 * (g_filter_sigma ** 2)))
        new_gaussian_img = cv2.filter2D(self.grayimg, -1, gaussian_kernal)
        cv2.imwrite("gaussian.jpg", new_gaussian_img)
        gaussian_img = cv2.imread("gaussian.jpg")
        height, width, bytesPerComponent = gaussian_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB, gaussian_img)
        QImg = QImage(gaussian_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_gaussian = QPixmap.fromImage(QImg)
        self.scene_3 = QGraphicsScene(self)
        self.item_3 = QGraphicsPixmapItem(self.pixmap_gaussian)
        self.scene_3.addItem(self.item_3)
        self.graphicsView_3.setScene(self.scene_3)

    ####### 高斯滤波##########
    def show_gaussian_smooth(self):
        g_filter_size = 3
        g_filter_sigma = 1.00
        gaussian_kernal = np.zeros([g_filter_size, g_filter_size])
        for i in range(g_filter_size):
            for j in range(g_filter_size):
                math_i_index = i - int((g_filter_size+1)/2)
                math_j_index = j - int((g_filter_size+1)/2)
                gaussian_kernal[i, j] = (1/(2*np.pi*g_filter_sigma**2))* \
                                        math.exp(-(math_i_index**2+math_j_index**2)/(2*(g_filter_sigma**2)))
        new_gaussian_img = cv2.filter2D(self.grayimg, -1, gaussian_kernal)
        cv2.imwrite("gaussian.jpg", new_gaussian_img)
        gaussian_img = cv2.imread("gaussian.jpg")
        height, width, bytesPerComponent = gaussian_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB, gaussian_img)
        QImg = QImage(gaussian_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_gaussian = QPixmap.fromImage(QImg)
        self.scene_3 = QGraphicsScene(self)
        self.item_3 = QGraphicsPixmapItem(self.pixmap_gaussian)
        self.scene_3.addItem(self.item_3)
        self.graphicsView_3.setScene(self.scene_3)

    ######### 自定义滤波器################
    def show_custom_dialog(self):
        dialog = Dialog()
        print(custom3)
        dialog.exec_()

    ############## 3*3 自定义滤波 ##############
    def show_custom_3_3(self):
        print(custom3)
        custom_kernal_3_3 = custom3
        custom3_img = cv2.filter2D(self.grayimg, -1, custom_kernal_3_3)
        cv2.imwrite("custom3.jpg", custom3_img)
        custom3_img = cv2.imread("custom3.jpg")
        height, width, bytesPerComponent = custom3_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(custom3_img, cv2.COLOR_BGR2RGB, custom3_img)
        QImg = QImage(custom3_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_custom3 = QPixmap.fromImage(QImg)
        self.scene_4 = QGraphicsScene(self)
        self.item_4 = QGraphicsPixmapItem(self.pixmap_custom3)
        self.scene_4.addItem(self.item_4)
        self.graphicsView_3.setScene(self.scene_4)

    ############## 5*5 自定义滤波 ##############
    def show_custom_5_5(self):
        print(custom5)
        custom_kernal_5_5 = custom5
        custom5_img = cv2.filter2D(self.grayimg, -1, custom_kernal_5_5)
        cv2.imwrite("custom5.jpg", custom5_img)
        custom5_img = cv2.imread("custom5.jpg")
        height, width, bytesPerComponent = custom5_img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(custom5_img, cv2.COLOR_BGR2RGB, custom5_img)
        QImg = QImage(custom5_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_custom5 = QPixmap.fromImage(QImg)
        self.scene_4 = QGraphicsScene(self)
        self.item_4 = QGraphicsPixmapItem(self.pixmap_custom5)
        self.scene_4.addItem(self.item_4)
        self.graphicsView_3.setScene(self.scene_4)


    ######### 设置形态学核的大小 ##############
    def set_kernal_size_1(self, kernel_value):
        self.size_flag_1 = 1
        self.kernal_size_1 = kernel_value
        # print(self.kernal_size_1)
        return self.kernal_size_1

    def set_kernal_size_2(self, kernel_value):
        self.size_flag_2 = 1
        self.kernal_size_2 = kernel_value
        return self.kernal_size_2

    ######## 把形态学核设置成椭圆 ########
    def setshape_ellipse(self):
        self.ellipse_flag = 1
        self.cross_flag = 0
        self.shape = cv2.MORPH_ELLIPSE
        return self.shape

    ######## 把形态学核设置成十字叉 ########
    def setshape_cross(self):
        self.cross_flag = 1
        self.ellipse_flag = 0
        self.shape = cv2.MORPH_CROSS
        return self.shape

    def setshape_rect(self):
        self.cross_flag = 0
        self.ellipse_flag = 0

    def binary_img(self, img_to_binary):
        rows, columns = img_to_binary.shape
        img_binary = np.zeros([rows, columns])
        for i in range(rows):
            for j in range(columns):
                if img_to_binary[i, j] > 30:
                    img_binary[i, j] = 255
                else:
                    img_binary[i, j] = 0
        return img_binary

    def erosion_operation(self, source_image):
        # decide the size_x of kernel
        kernel_size_x_string = self.lineEdit_sizex.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3

        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx.text()
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
                anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
        # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        img_original = source_image
        rows, columns = img_original.shape
        # decide the kernel shape
        shape = cv2.MORPH_RECT
        if self.cross_flag == 1:
            shape = self.setshape_cross()
        if self.ellipse_flag == 1:
            shape = self.setshape_ellipse()
        self.mor_kernel = cv2.getStructuringElement(shape, (kernel_size_x, kernel_size_y))
        I_errosed2 =cv2.erode(source_image, self.mor_kernel)
        kernel_rows = self.mor_kernel.shape[0]
        kernel_columns = self.mor_kernel.shape[1]
        I_errosed = np.zeros([rows, columns])
        temp_min = []
        for i in range(0+anchor_x, rows - kernel_rows-anchor_x):
            for j in range(0+anchor_y, columns - kernel_columns-anchor_y):
                img_box = img_original[i-anchor_x:i + kernel_rows-anchor_x, j-anchor_y:j + kernel_columns-anchor_y]
                for m in range(kernel_rows):
                    for n in range(kernel_columns):
                        if self.mor_kernel[m, n] == 1:
                            temp_min.append(img_box[m, n])
                I_errosed[i, j] = min(temp_min)
                temp_min = []
        return I_errosed, I_errosed2

    ############# 腐蚀 #######
    def my_erosion(self):
        image_path = self.my_file_path[0]
        img_original = cv2.imread(image_path, flags=0)
        source_image = img_original
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        I_errosion, I_errosion2 = self.erosion_operation(source_image=source_image)
        cv2.imwrite("erosion.jpg", I_errosion2)
        erosion_img = cv2.imread("erosion.jpg")
        self.imshow_ori(erosion_img, 3)


    def dilation_operation(self, source_image):
        # decide the size_x of kernel
        kernel_size_x_string = self.lineEdit_sizex.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3

        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx.text()
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
               anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
        # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        img_original = source_image
        rows, columns = img_original.shape
        # decide the kernel shape
        shape = cv2.MORPH_RECT
        if self.cross_flag == 1:
            shape = self.setshape_cross()
        if self.ellipse_flag == 1:
            shape = self.setshape_ellipse()
        self.mor_kernel = cv2.getStructuringElement(shape, (kernel_size_x, kernel_size_y))
        I_dilated2 = cv2.dilate(source_image, self.mor_kernel)
        kernel_rows = self.mor_kernel.shape[0]
        kernel_columns = self.mor_kernel.shape[1]
        I_dilated = np.zeros([rows, columns])
        temp_max = []
        for i in range(0+anchor_x, rows - kernel_rows-anchor_x):
            for j in range(0+anchor_y, columns - kernel_columns-anchor_y):
                img_box = img_original[i-anchor_x:i + kernel_rows-anchor_x, j-anchor_y:j + kernel_columns-anchor_y]
                for m in range(kernel_rows):
                    for n in range(kernel_columns):
                        if self.mor_kernel[m, n] == 1:
                            temp_max.append(img_box[m, n])
                I_dilated[i, j] = max(temp_max)
                temp_max = []
        return I_dilated, I_dilated2


    def my_dilation(self):
        image_path = self.my_file_path[0]
        original_img = cv2.imread(image_path, flags=0)
        source_image = original_img
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        _, I_dilation = self.dilation_operation(source_image=source_image)
        cv2.imwrite("dilation.jpg", I_dilation)
        dilation_img = cv2.imread("dilation.jpg")
        self.imshow_ori(img_to_show=dilation_img, viewer_index=4)

    ########### 开操作 #############
    def my_opening(self):
        image_path = self.my_file_path[0]
        original_img = cv2.imread(image_path, flags=0)
        source_image = original_img
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        I_erosed, I_erosed2 = self.erosion_operation(source_image=source_image)
        _, I_opened = self.dilation_operation(source_image=I_erosed2)
        cv2.imwrite("open.jpg", I_opened)
        open_img = cv2.imread("open.jpg")
        self.imshow_ori(img_to_show=open_img, viewer_index=3)

    ############### 闭操作 ##############
    def my_closing(self):
        image_path = self.my_file_path[0]
        original_img = cv2.imread(image_path, flags=0)
        source_image = original_img
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        _, I_dilated = self.dilation_operation(source_image=source_image)
        _, I_closed = self.erosion_operation(source_image=I_dilated)
        cv2.imwrite("close.jpg", I_closed)
        closed_img = cv2.imread("close.jpg")
        self.imshow_ori(img_to_show=closed_img, viewer_index=4)


    def find_matrix_max_value(self, input_matrix):
        rows, columns = input_matrix.shape
        temp_max = []
        for i in range(rows):
            temp_max.append(max(input_matrix[i]))
        return max(temp_max)

    ########## distance transform #########
    def distance_transform(self):
        image_path = self.my_file_path[0]
        original_img = cv2.imread(image_path, flags=0)
        source_image = original_img
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        img_max_value = self.find_matrix_max_value(source_image)
        self.I_show = np.zeros([source_image.shape[0], source_image.shape[1]])
        erosion_count = 0
        source_image = source_image * 1.00
        while(img_max_value ):
            I_erosed, I_erosed2 = self.erosion_operation(source_image)
            I_changed_1 = source_image - I_erosed2
            I_changed_1 = (erosion_count+1)/255*I_changed_1
            self.I_show += I_changed_1
            I_show2 = self.I_show*5
            source_image = I_erosed2
            erosion_count += 1
            self.textBrowser_2.append('dst:'+str(erosion_count))
            img_max_value = self.find_matrix_max_value(I_erosed2)
            name = 'distance_transform.jpg'
            cv2.imwrite(name, I_show2)
            distance_transform_img = cv2.imread(name)
            self.graphicsView_2.setUpdatesEnabled(True)
            self.imshow_ori(img_to_show=distance_transform_img, viewer_index=2)
            self.graphicsView_2.repaint()
        I_show_finally = self.I_show/erosion_count*255
        self.distance_name = 'distance_transform.jpg'
        cv2.imwrite(self.distance_name, I_show_finally)
        distance_transform_img = cv2.imread(self.distance_name)
        self.graphicsView_2.setUpdatesEnabled(True)
        self.imshow_ori(img_to_show=distance_transform_img, viewer_index=2)
        self.graphicsView_2.repaint()


    ######################################################
    # skeleton
    ##############################################
    def skeleton(self):
        image_path = self.my_file_path[0]
        original_img = cv2.imread(image_path, flags=0)
        source_image = original_img
        if self.binary_flag == 1:
            source_image = self.grayimg_manual
        kernel_size_x_string = self.lineEdit_sizex.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3

        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx.text()
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
                anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
        # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        img_original = source_image
        rows, columns = img_original.shape
        # decide the kernel shape
        shape = cv2.MORPH_RECT
        if self.cross_flag == 1:
            shape = self.setshape_cross()
        if self.ellipse_flag == 1:
            shape = self.setshape_ellipse()
        mor_kernel = cv2.getStructuringElement(shape, (kernel_size_x, kernel_size_y))
        img_max_value = self.find_matrix_max_value(source_image)
        I_show = np.zeros([source_image.shape[0], source_image.shape[1]])
        I_show_temp = np.zeros([source_image.shape[0], source_image.shape[1]])
        source_image = source_image * 1.00
        self.skeleton_count = 0
        while (img_max_value):
            I_erosed = cv2.erode(source_image, mor_kernel, anchor=(anchor_x, anchor_y))
            I_erosed_erosed = cv2.erode(I_erosed, mor_kernel, anchor=(anchor_x, anchor_y))
            I_erosed_opend = cv2.dilate(I_erosed_erosed, mor_kernel, anchor=(anchor_x, anchor_y))
            # _, I_erosed = self.erosion_operation(source_image)
            # _, I_erosed_erosed = self.erosion_operation(I_erosed)
            # _, I_erosed_opend = self.dilation_operation(I_erosed_erosed)
            I_show_temp += (I_erosed-I_erosed_opend)/255*(self.skeleton_count+1)
            I_show += I_erosed-I_erosed_opend
            source_image = I_erosed
            name = 'skeleton.jpg'
            cv2.imwrite(name, I_show)
            skeleton = cv2.imread(name)
            self.graphicsView_3.setUpdatesEnabled(True)
            self.imshow_ori(img_to_show=skeleton, viewer_index=3)
            self.graphicsView_3.repaint()
            img_max_value = self.find_matrix_max_value(I_erosed)
            self.skeleton_count += 1
            print(self.skeleton_count)
        cv2.imwrite('skeleton_temp.jpg', I_show_temp)


    #################### skeleton restoration ###########################
    def skeleton_restoration(self):
        I_skeletoned = cv2.imread('skeleton_temp.jpg', flags=0)
        rows, columns = I_skeletoned.shape
        # I_show = np.zeros([I_skeletoned.shape[0], I_skeletoned.shape[1]])
        kernel_size_x_string = self.lineEdit_sizex.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3
        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx.text()
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
                anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
        # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        # decide the kernel shape
        while(self.skeleton_count > 0):
            for i in range(kernel_size_x, rows-kernel_size_x):
                for j in range(kernel_size_y, columns-kernel_size_y):
                    if I_skeletoned[i, j] == self.skeleton_count:
                        for m in range(0-anchor_x, kernel_size_x-anchor_x):
                            for n in range(0-anchor_y, kernel_size_y-anchor_y):
                                if I_skeletoned[i+m, j+n]<self.skeleton_count:
                                    I_skeletoned[i+m, j+n] = self.skeleton_count-1

            I_show = (I_skeletoned)*5
            self.skeleton_count -= 1
            name = 'skeleton_restoration.jpg'
            cv2.imwrite(name, I_show)
            skeleton_restoration = cv2.imread(name)
            self.graphicsView_4.setUpdatesEnabled(True)
            self.imshow_ori(img_to_show=skeleton_restoration, viewer_index=4)
            self.graphicsView_4.repaint()
        I_show_sr = np.zeros([rows, columns])
        for i in range(rows):
            for j in range(columns):
                if I_show[i, j] >0:
                    I_show_sr[i, j] = 255
        cv2.imwrite(name, I_show_sr)
        skeleton_restoration = cv2.imread(name)
        self.graphicsView_4.setUpdatesEnabled(True)
        self.imshow_ori(img_to_show=skeleton_restoration, viewer_index=4)
        self.graphicsView_4.repaint()



    ########################## grayscale erosion ##################
    def erosion_gray_operation(self, source_image):
        # decide the size_x of kernel
        kernel_size_x_string = self.lineEdit_sizex_gray.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3

        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey_gray.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx_gray.text()
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
                anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
            # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy_gray.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        img_original = source_image
        rows, columns = img_original.shape
        # decide the kernel shape
        shape = cv2.MORPH_RECT
        if self.cross_flag == 1:
            shape = self.setshape_cross()
        if self.ellipse_flag == 1:
            shape = self.setshape_ellipse()
        mor_kernel = cv2.getStructuringElement(shape, (kernel_size_x, kernel_size_y))
        kernel_rows = mor_kernel.shape[0]
        kernel_columns = mor_kernel.shape[1]
        I_errosed = np.zeros([rows, columns])
        temp_min = []
        for i in range(0+anchor_x, rows - kernel_rows-anchor_x):
            for j in range(0+anchor_y, columns - kernel_columns-anchor_y):
                img_box = img_original[i-anchor_x:i + kernel_rows-anchor_x, j-anchor_y:j + kernel_columns-anchor_y]
                for m in range(kernel_rows):
                    for n in range(kernel_columns):
                        if mor_kernel[m, n] == 1:
                            temp_min.append(img_box[m, n])
                I_errosed[i, j] = min(temp_min)
                temp_min = []
        return I_errosed


    def erosion_gray(self):
        I_input = cv2.imread(self.my_file_path[0], flags=0)
        I_errosed = self.erosion_gray_operation(source_image=I_input)
        cv2.imwrite("erosed_gray.jpg", I_errosed)
        I_erosed_gray = cv2.imread("erosed_gray.jpg")
        self.imshow_ori(img_to_show=I_erosed_gray, viewer_index=2)

    ########################## grayscale dilation ##################
    def dilation_gray_operation(self, source_image):
        # decide the size_x of kernel
        kernel_size_x_string = self.lineEdit_sizex_gray.text()
        if str.isdigit(kernel_size_x_string):
            kernel_size_x = int(kernel_size_x_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_x = 3

        # decide the size_y of kernel
        kernel_size_y_string = self.lineEdit_sizey_gray.text()
        if str.isdigit(kernel_size_y_string):
            kernel_size_y = int(kernel_size_y_string)
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            kernel_size_y = 3

        # decide the anchor_x of kernel
        anchor_x_string = self.lineEdit_originx_gray.text()

        print(anchor_x_string)
        if str.isdigit(anchor_x_string):
            anchor_x = int(anchor_x_string)
            if 0 <= anchor_x < kernel_size_x:
                anchor_x = anchor_x
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_x = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_x = 1
        print(anchor_x)
        # decide the anchor_y of kernel
        anchor_y_string = self.lineEdit_originy_gray.text()
        if str.isdigit(anchor_y_string):
            anchor_y = int(anchor_y_string)
            if 0 <= anchor_y < kernel_size_y:
                anchor_y = anchor_y
            else:
                QMessageBox.information(self, 'information',
                                        'please input a natural number between '
                                        '0 and kernel size x')
                anchor_y = 1
        else:
            QMessageBox.information(self, 'information',
                                    'please input a natural number')
            anchor_y = 1
        img_original = source_image
        rows, columns = img_original.shape
        # decide the kernel shape
        shape = cv2.MORPH_RECT
        if self.cross_flag == 1:
            shape = self.setshape_cross()
        if self.ellipse_flag == 1:
            shape = self.setshape_ellipse()
        mor_kernel = cv2.getStructuringElement(shape, (kernel_size_x, kernel_size_y))
        kernel_rows = mor_kernel.shape[0]
        kernel_columns = mor_kernel.shape[1]
        I_errosed = np.zeros([rows, columns])
        temp_max = []
        for i in range(0+anchor_x, rows - kernel_rows-anchor_x):
            for j in range(0+anchor_y, columns - kernel_columns-anchor_y):
                img_box = img_original[i-anchor_x:i + kernel_rows-anchor_x, j-anchor_y:j + kernel_columns-anchor_y]
                for m in range(kernel_rows):
                    for n in range(kernel_columns):
                        if mor_kernel[m, n] == 1:
                            temp_max.append(img_box[m, n])
                I_errosed[i, j] = max(temp_max)
                temp_max = []
        return I_errosed

    def dilation_gray(self):
        I_input = cv2.imread(self.my_file_path[0], flags=0)
        I_dilated_gray = self.dilation_gray_operation(source_image=I_input)
        cv2.imwrite("dilated_gray.jpg", I_dilated_gray)
        I_erosed_gray = cv2.imread("dilated_gray.jpg")
        self.imshow_ori(img_to_show=I_erosed_gray, viewer_index=2)

    def closing_gray(self):
        I_input = cv2.imread(self.my_file_path[0], flags=0)
        I_dilated_gray = self.dilation_gray_operation(source_image=I_input)
        I_closed_gray = self.erosion_gray_operation(source_image=I_dilated_gray)
        cv2.imwrite("closing_gray.jpg", I_closed_gray)
        I_closed_gray = cv2.imread("closing_gray.jpg")
        self.imshow_ori(img_to_show=I_closed_gray, viewer_index=4)

    def opening_gray(self):
        I_input = cv2.imread(self.my_file_path[0], flags=0)
        I_erosed_gray = self.erosion_gray_operation(source_image=I_input)
        I_opened_gray = self.dilation_gray_operation(source_image=I_erosed_gray)
        cv2.imwrite("closing_gray.jpg", I_opened_gray)
        I_opened_gray = cv2.imread("closing_gray.jpg")
        self.imshow_ori(img_to_show=I_opened_gray, viewer_index=3)

    def set_edge_type_st(self):
        self.edge_ex_flag = 0
        self.edge_in_flag = 0

    def set_edge_type_ex(self):
        self.edge_ex_flag = 1
        self.edge_in_flag = 0

    def set_edge_type_in(self):
        self.edge_ex_flag = 0
        self.edge_in_flag = 1

    def set_gradient_type_st(self):
        self.gradient_ex_flag = 0
        self.gradient_in_flag = 0

    def set_gradient_type_ex(self):
        self.gradient_ex_flag = 1
        self.gradient_in_flag = 0

    def set_gradient_type_in(self):
        self.gradient_ex_flag = 0
        self.gradient_in_flag = 1

    def set_reconstruction_type_cbr(self):
        self.obr_flag = 0

    def set_reconstruction_type_obr(self):
        self.obr_flag = 1

    def mor_egde_gray(self):
        input_image = cv2.imread(self.my_file_path[0], flags=0)
        if self.binary_flag == 1:
            input_image = self.grayimg_manual
        I_erosed = self.erosion_gray_operation(source_image=input_image)
        I_dilated = self.dilation_gray_operation(source_image=input_image)
        mor_edge = I_dilated - I_erosed
        if self.edge_ex_flag == 1:
            mor_edge = I_dilated - input_image
        if self.edge_in_flag == 1:
            mor_edge = input_image - I_erosed
        cv2.imwrite('mor_edge.jpg', mor_edge)
        I_show = cv2.imread('mor_edge.jpg')
        self.imshow_ori(img_to_show=I_show, viewer_index=2)

    def mor_gradient_gray(self):
        input_image = cv2.imread(self.my_file_path[0], flags=0)
        if self.binary_flag == 1:
            input_image = self.grayimg_manual
        I_erosed = self.erosion_gray_operation(source_image=input_image)
        I_dilated = self.dilation_gray_operation(source_image=input_image)
        mor_gradient = (I_dilated - I_erosed)/2
        if self.gradient_ex_flag == 1:
            mor_gradient = (I_dilated - input_image)/2
        if self.gradient_in_flag == 1:
            mor_gradient = (input_image -I_erosed)/2
        cv2.imwrite('mor_gradient.jpg', mor_gradient)
        I_show = cv2.imread('mor_gradient.jpg')
        self.imshow_ori(img_to_show=I_show, viewer_index=2)

    def cbr(self):
        input_image = cv2.imread(self.my_file_path[0], flags=0)
        if self.binary_flag == 1:
            input_image = self.grayimg_manual
        marker_tem_erosed = self.erosion_gray_operation(source_image=input_image)
        marker = self.dilation_gray_operation(source_image=marker_tem_erosed)
        name_1 = 'cbr_marker.jpg'
        cv2.imwrite(name_1, marker)
        I_show = cv2.imread(name_1)
        self.imshow_ori(img_to_show=I_show, viewer_index=3)
        # self.imshow_ori(img_to_show=marker, viewer_index=4)
        error = 1
        I_reconstruction_initial = marker
        I_reconstruction = marker
        while(error):
            I_dilated = self.dilation_gray_operation(source_image=I_reconstruction_initial)
            for i in range(input_image.shape[0]):
                for j in range(input_image.shape[1]):
                    if I_dilated[i, j] <= input_image[i, j]:
                        I_reconstruction[i, j] = I_dilated[i,j]
                    else:
                        I_reconstruction[i, j] = 0
            error = I_reconstruction - I_reconstruction_initial
            error_max = self.find_matrix_max_value(error)
            error = error_max
            I_reconstruction_initial = I_reconstruction
            name = 'cbr.jpg'
            cv2.imwrite(name, I_reconstruction)
            I_show = cv2.imread(name)
            self.graphicsView_4.setUpdatesEnabled(True)
            self.imshow_ori(img_to_show=I_show, viewer_index=4)
            self.graphicsView_4.repaint()
            print(error)















from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog
from custom import  Ui_Dialog

custom3 = np.zeros([3, 3])
custom5 = np.zeros([5, 5])
class Dialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("icon_mainwindow.png"))

    ############# 传递设计 5*5 的核##########
    @pyqtSlot()
    def on_pushButton_clicked(self):
        custom5[0, 0] = float(self.lineEdit_0_0.text())
        custom5[0, 1] = float(self.lineEdit_0_1.text())
        custom5[0, 2] = float(self.lineEdit_0_2.text())
        custom5[0, 3] = float(self.lineEdit_0_3.text())
        custom5[0, 4] = float(self.lineEdit_0_4.text())
        custom5[1, 0] = float(self.lineEdit_1_0.text())
        custom5[1, 1] = float(self.lineEdit_1_1.text())
        custom5[1, 2] = float(self.lineEdit_1_2.text())
        custom5[1, 3] = float(self.lineEdit_1_3.text())
        custom5[1, 4] = float(self.lineEdit_1_4.text())
        custom5[2, 0] = float(self.lineEdit_2_0.text())
        custom5[2, 1] = float(self.lineEdit_2_1.text())
        custom5[2, 2] = float(self.lineEdit_2_2.text())
        custom5[2, 3] = float(self.lineEdit_2_3.text())
        custom5[2, 4] = float(self.lineEdit_2_4.text())
        custom5[3, 0] = float(self.lineEdit_3_0.text())
        custom5[3, 1] = float(self.lineEdit_3_1.text())
        custom5[3, 2] = float(self.lineEdit_3_2.text())
        custom5[3, 3] = float(self.lineEdit_3_3.text())
        custom5[3, 4] = float(self.lineEdit_3_4.text())
        custom5[4, 0] = float(self.lineEdit_4_0.text())
        custom5[4, 1] = float(self.lineEdit_4_1.text())
        custom5[4, 2] = float(self.lineEdit_4_2.text())
        custom5[4, 3] = float(self.lineEdit_4_3.text())
        custom5[4, 4] = float(self.lineEdit_4_4.text())

    ############# 传递设计 5*5 的核##########
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        ##### 竟然不能循环识别？？？######
        # for i in range(1, 4):
        #     for j in range(1, 4):
        #         custom3[i, j] = self.lineEdit_i_j.text()
        custom3[0,0] = float(self.lineEdit_1_1.text())
        custom3[0,1] = float(self.lineEdit_1_2.text())
        custom3[0,2] = float(self.lineEdit_1_3.text())
        custom3[1,0] = float(self.lineEdit_2_1.text())
        custom3[1,1] = float(self.lineEdit_2_2.text())
        custom3[1,2] = float(self.lineEdit_2_3.text())
        custom3[2,0] = float(self.lineEdit_3_1.text())
        custom3[2,1] = float(self.lineEdit_3_2.text())
        custom3[2,2] = float(self.lineEdit_3_3.text())


    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.close()




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    splash = QSplashScreen(QPixmap( "cover_small.png"))    #不要加路径
    splash.show()
    for i in range(0, 100, 10):
        splash.showMessage("         The program is initiating %d " % i,Qt.AlignLeft|Qt.AlignVCenter)
    app.processEvents()  # 使程序在显示启动画面的同时仍能响应鼠标其他事件
    ui = MainWindow()
    ui.show()
    splash.finish(ui)
    sys.exit(app.exec_())
