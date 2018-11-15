#ZJP
#icon.py  22:36
'''
this example show an icon in the titlebar of the window
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon


class Examlpe(QWidget):
    def __init__(self):
        super().__init__()     # ? return the parent object and call its constructor调用父类的析构函数 必须
        self.initUI()          # 实体属性

    def initUI(self):
        self.setGeometry(300, 300, 300, 220)  # 控制位置和大小，前面两个控制位置，后面两个控制大小
        self.setWindowTitle('icon')
        self.setWindowIcon(QIcon('black.jpg')) # set application icon,create a object icon, receive the path to our icon to be displayed

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Examlpe()
    sys.exit(app.exec_())

