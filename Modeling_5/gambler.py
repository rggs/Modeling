#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:41:14 2020

@author: ryanswope
"""
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


def Roulette(capital,stakes,ret,tries):
    size = int(ret*capital/stakes)+1
    print(str(size))
    mat=np.zeros((size,size))
    
    mat[0,0]=1
    mat[-1,-1]=1
    
    p=18/37
    
    col=1
    row0=0
    row2=2
    
    while col<size-1 and row0<size:
        mat[row0,col]=1-p
        col+=1
        row0+=1
        #print('p-1 '+str(col)+', '+str(row0))
        
        
    col=1
    while col<size-1  and row2<size:
        mat[row2,col]=p
        row2+=1
        col+=1
        #print('p '+ str(col)+', '+str(row2))
        
    state=np.zeros(size)
    state[int((size-1)/ret)]=1
    
    final_mat=np.linalg.matrix_power(mat,tries)
    return np.matmul(final_mat,state),final_mat

result, mat =Roulette(1000,100,3,100000)

#print(result)
'''
lose=[]
win=[]
tries=[]
for j in range(1,10000):
    state, mat=Roulette(1000,100,1.5,j)
    tries.append(j)
    lose.append(state[0])
    win.append(state[-1])
    print(j)
    
    
f1=plt.figure()
ax1=f1.add_subplot(111)

ax1.plot(tries,win,label="Winning",color='black')
ax1.plot(tries,lose,label='Losing',color='red')
ax1.set_xlabel('Tries')
ax1.set_ylabel('Probability')
ax1.legend()
f1.tight_layout()
'''



'''
def city(N,A):
    
    city_grid=np.zeros((A,A))
    
    for i in range(N):
        placed=False
        while (placed==False):
            xloc=np.random.randint(0,high=A)
            yloc=np.random.randint(0,high=A)
            
            if city_grid[xloc,yloc]==0:
                city_grid[xloc,yloc]=1
                placed=True
                
    return city_grid
'''

# def epidemic(N,A,steps):    
#      size = 2
#      print(str(size))
#      mat=np.zeros((size,size))  
#      sigma=N/A
#      mat[0,0]=1-sigma
#      mat[1,0]=sigma
#      mat[-1,-1]=1  
#      state=[N,0]
#      final_mat=np.linalg.matrix_power(mat,steps)
#      return np.matmul(final_mat,state)
# steps=[]
# sick=[]
# healthy=[]
# for j in range(1000):
#     steps.append(j)
#     plague=epidemic(100,10000,j)
#     sick.append(plague[0])
#     healthy.append(plague[1])
# f2=plt.figure()
# ax2=f2.add_subplot(111)
# ax2.plot(steps,sick,label='Sick',color='red')
# ax2.plot(steps,healthy,label='Healthy',color='green')
# ax2.set_xlabel('Steps')
# ax2.set_ylabel('Population')
# ax2.set_title('Disease Spread')
# ax2.legend()

# Python code for 2D random walk. 
# import numpy 
# import random 

# defining the number of steps 
# n = 1000

#creating two array for containing x and y coordinate 
#of size equals to the number of size and filled up with 0's 
# x = numpy.zeros(n) 
# y = numpy.zeros(n) 

# filling the coordinates with random variables 
# for i in range(1, n): 
# 	val = random.randint(1, 4) 
# 	if val == 1: 
# 		x[i] = x[i - 1] + 1
# 		y[i] = y[i - 1] 
# 	elif val == 2: 
# 		x[i] = x[i - 1] - 1
# 		y[i] = y[i - 1] 
# 	elif val == 3: 
# 		x[i] = x[i - 1] 
# 		y[i] = y[i - 1] + 1
# 	else: 
# 		x[i] = x[i - 1] 
# 		y[i] = y[i - 1] - 1
	

# plotting stuff: 
# f3=plt.figure()
# ax3=f3.add_subplot(111)
# ax3.set_title("Random Walk ($n = " + str(n) + "$ steps)") 
# ax3.plot(x, y) 
#plt.savefig("rand_walk"+str(n)+".png",bbox_inches="tight",dpi=600) 


 





    

        
        