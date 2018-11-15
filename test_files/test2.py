#ZJP
#test2.py  16:43

import py_compile

from test_files import test

print(test.hello.__doc__)
py_compile.compile('show_histogram.py')

