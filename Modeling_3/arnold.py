#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:27:45 2020

@author: ryanswope
"""

import numpy as np
import scipy
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

donkey=mpimg.imread('/Users/ryanswope/PythonDocs/Arnold/kong.png')
donkey2=np.array(Image.open('/Users/ryanswope/PythonDocs/Arnold/kong_small.png'))

donkey_orig = donkey2.copy()

xsize=donkey2.shape[1]
ysize=donkey2.shape[0]

test=[]
'''
for i in range(xsize):
    row=[]
    for j in range(ysize):
        row.append([i,j])
    test.append(row)
'''
x,y=np.meshgrid(range(xsize),range(ysize))

        
def arnold(x,y):
    xnew=(2*x+y)%xsize
    ynew=(x+y)%ysize
    
    return xnew,ynew
    '''
    a=np.array([[2,1],[1,1]])
    new_pic=[]
    for k in flop:
        for j in k:
            new=np.matmul(a,j)
            new_pic.append(new)
     '''
'''
def speed(iterations):
    length=[]
    for j in range(1,iterations):
        x,y=np.meshgrid(range(j),range(j))
        i=0
        X,Y=arnold(x,y,j,j)
        while(not np.array_equal(X,x) and not np.array_equal(Y,y)):
            X,Y=arnold(X,Y,j,j)
            i+=1
            print(str(j), 'Iterations: '+str(i))
        length.append([j,i])
    return length
    '''

#Try plotting i as a function of n

i=100000

repeat = False
one_more = False

X,Y=arnold(x,y)

while not repeat:
    
    if repeat:
        one_more = True
    X,Y=arnold(X,Y)
    #repeat = np.array_equal(X,x) and np.array_equal(Y,y)
    
    donkey2=donkey2[X,Y]
    
    repeat = np.array_equal(donkey2, donkey_orig)

    print('Iterations: '+str(i-99999))
    if i%1==0:
        filename='/Users/ryanswope/PythonDocs/Arnold/kong_small_gif/konggif'+str(i)+'.png'
        im = Image.fromarray(donkey2)
        im.save(filename)
        #plt.savefig(filename,)
    i+=1