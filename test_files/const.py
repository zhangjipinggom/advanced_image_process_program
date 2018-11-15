#ZJP
#const.py  20:50

# a module define the constant variance

class _const:
    class ConstError(TypeError):pass
    def __setattr__(self, key, value):  # 实例化时会自动调用
        # if self.__dict__.has_key(key):
        #     raise(self.ConstError,"can't rebind const %s " % key)
        self.__dict__[key] = value


import sys
sys.modules[__name__] = _const()    # 把const类注册到sys.module这个全局字典中





