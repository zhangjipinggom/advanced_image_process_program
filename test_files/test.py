#ZJP
#test.py  16:15
'''___doc__module'''
import sys
import copy

###   ergodic way 1  #############
list_tuple = []
tuple_num = ((0, 0), (1, 1),(2, 2))
for i in range(len(tuple_num)):
    print("tumple[%d] :" % i,"")
    for j in range(len(tuple_num[i])):
        print("tuple[[%d][%d]]"%(i,j),"",tuple_num[i][j])
    print()


list_tuple = tuple_num
list_num = [1, 2, 3, 4, 5]
list_num2 = [6]
list_num.append(list_tuple)
list_num.insert(1, '6')
list_num.remove(3)
print(list_num)


new_list = list_num.extend(list_tuple)
new_list = list_num*2
print(new_list)




dictionary = {1:(1, 1), 2:(2, 2), 3:(3, 3)}

del(dictionary[1])
print(dictionary)
print(dictionary.pop(2))

dictionary2 = {'a': 'apple', 'b': 'banana'}
for k in dictionary2:
    print(k, dictionary2[k])
    print("key is %s;" % k, "value is %s."% dictionary2[k])

for i,j in dictionary2.items():
    print("dictionary2[%s]" % i, j)

print(dictionary2.get('e'))
#dictionary2.update(dictionary)
dictionary2.setdefault('e','egg')
print(dictionary2)
#print(sorted(dictionary2.items(),key=lambda d:d[0]))
dictionary2 = copy.deepcopy(dictionary)
print(dictionary2)


words = 'inconspicuous'
print(words.center(40, "*"))




