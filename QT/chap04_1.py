#ZJP
#chap04_1.py  11:15

import sys
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)   # 提供访问全局信息的能力，传递了命令行参数？

try:
    due = QTime.currentTime()
    message = "Alert"
    if len(sys.argv)<2:
        raise ValueError
    hours, mins = sys.argv[1].split(":")
    due = QTime(int(hours),int(mins))
    if not due.isValid():
        raise ValueError
    if len(sys.argv)>2:
        message = " ".join(sys.argv[2:])
except ValueError:
    message = "Usage: alert.pyw HH:MM[optional message]"