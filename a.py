import numpy as np
import time

num = 100000000
my_array = np.arange(num)     # NumPy 陣列，值：[0, num)
my_list = list(range(num))    # Python 串列，值：[0, num)

startTime = time.time()
my_array2 = my_array * 2
endTime = time.time()
a1 = endTime - startTime
print('NumPy array multiplication time:', a1)

startTime = time.time()
my_list2 = [x*2 for x in my_list]
endTime = time.time()
a2 = endTime - startTime
print('Python list comprehension time:', a2)
print(a2/a1)