#ZJP
#graph_include_start.py  22:43

#ZJP
#dialog.py  22:06

#ZJP
#text_browes.py  21:27

#ZJP
#test2.py  20:18
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow,QMessageBox,QInputDialog,QSplashScreen,QFileDialog
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
import webbrowser
import time

from  ui_graph import Ui_MainWindow
class MainWindow(QMainWindow,Ui_MainWindow):
    """
    由括号中的两类继承而来
    """
    def __init__(self,parent = None):
        '''
        constructer
        :param parent:
        '''
        QMainWindow.__init__(self,parent)
        self.setupUi(self)
        self.graphicsView.mousePressEvent = self.my_clicked
        time.sleep(2)

    def my_clicked(self,event):
        print('haha')
        webbrowser.open("www.baidu.com")

    #@pyqtSlot()
    @pyqtSlot()
    def on_pushButton_clicked(self):
        text_string =  self.textEdit.toPlainText()
        #self.textBrowser.setText(text_string)
        brower_text = self.textBrowser.toPlainText()
        self.textBrowser.append(text_string)
        print(brower_text)

    def on_pushButton2_clicked(self):
        #self.textEdit.setText('')
        self.textBrowser.property()
        self.textBrowser.clear('')

    def on_pushButton3_clicked(self):
        #my_button = QMessageBox.information(self,'hhah','提示')
        my_str,ok = QInputDialog.getText(self)

    def on_actionOpen_triggered(self):
        print("打开")
        my_file_path = QFileDialog.getOpenFileName(self, "open file", '/')
        print(my_file_path)
        # if my_file_path[0][-5:] == '.docx':
        #     from win32com import clent as wc
        #     word = wc.Dispatch('Word.Application')
        #     word.Visible = 1
        #     my_worddoc = word.Ducuments.Open(my_file_path[0].replace('/', '\\'))
        #     print(my_worddoc.Range)
        #     my_worddoc.Close()

        # f = open(my_file_path[0])
        # my_data = f.read()
        # f.close()
        # self.textEdit.setText(my_data)



    def on_actionClose_triggered(self):
        sys.exit()

    def on_actionAbout_triggered(self):
        QMessageBox.about(self,'关于这个例子','这是QT做的第一个例子')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    splash = QSplashScreen(QPixmap( "pic.jpg"))    #不要加路径
    splash.show()
    splash.showMessage("hahahah",Qt.AlignCenter, Qt.red)
    app.processEvents()
    ui = MainWindow()
    ui.show()
    splash.finish(ui)
    sys.exit(app.exec_())