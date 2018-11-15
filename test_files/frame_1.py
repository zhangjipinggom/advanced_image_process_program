#ZJP
#frame_1.py  21:50

import sys
from PyQt5.QtWidgets import QApplication,QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)  # sys.argv is a parameter list，必须创建一个应用程序对象，某则不会显示
    w = QWidget()   # 用户界面的基类
    w.resize(250,150)
    w.move(300, 300)
    w.setWindowTitle('zjp come on')
    w.show()

    sys.exit(app.exec())


