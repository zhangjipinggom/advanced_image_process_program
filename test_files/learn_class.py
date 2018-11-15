#ZJP
#learn_class.py  19:54

class Fruit:
    initial_price = 0
    __zone = 'China'

    def __init__(self):   #构造函数
        self.size = 10
        color = 'red'
        self.__taste = 'sweet'

class My_fruit(Fruit):
    fruit_name = 'orange'
    __orange_price = 0

    @staticmethod
    def get_color():
        print("color = orange")

    def get_price(self):
        print(self.__orange_price)






apple = Fruit()
print(apple.initial_price)
print(apple._Fruit__zone)   #访问私有属性的特殊表达
orange = My_fruit()
print(orange.fruit_name)
My_fruit.get_color()
orange.get_color()
orange.get_price()
