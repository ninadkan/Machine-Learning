# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:44:23 2018

@author: ninadk
"""

compoundinterest = 1.1
type(compoundinterest)

# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

print("I said " + ("Hey " * 2) + "Hey!")
print(True+ False)
print("I can add integers, like " + str(5) + " to strings.")
#error
print("The correct answer to this multiple choice exercise is answer number " + 2)
# =========================================================================================

# python list. [, ] ; List is collection of values. Its perfectly possible for list to 
# have different data types, and contain lists themselves. Main list contains four sublists. 
# creating a copy of the list, otherwise the first reference is copied. 
# y = list(x)
# y = x[:]
# del(x[-3])
#?max or help(max) to call help from the ipython console
# list.index, list.count; 
# list.append changges the list. so does remove, reverse
# from numpy import array. now we are using only simple module so not calling importing 
# Numpy --> list is slow. NumPy --> numeric Python. 
# NumPy can perform calculations on entire arrays. 
# arrays contains values of single arrays; booleans and floats would be converted to strings.
# True is converted to 1 and false to 0 when doing coercsion 
# array.shape will give you the dimensions of the array. 
# np_2d[0][2] is same as np_rd[0,2]; or 
# np.mean; np.median; 
# np.std --> standard deviation. np.corrcoef , correlation between two datasets