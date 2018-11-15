#ZJP
#simple.py  22:11

'''
in this example,we create a simple window in PyQt5
'''

import sys
from PyQt5.QtWidgets import QApplication,QWidget


if __name__ == "__main__":
    app = QApplication(sys.argv)   #sys.argv = ['E:/advanced_image_processing/simple.py']. it is a way we can control the
                                   #startup of our scripts.
                                   # every PyQt5 application must create an application object.
    w = QWidget()                   # widget is teh base class for all user interface. a widget with no parent called windows
    w.resize(250,150)
    w.move(300,300)
    w.setWindowTitle("simple")
    w.show()

    sys.exit(app.exec_())


