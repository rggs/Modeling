#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:47:37 2020

@author: rachelprice
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

t = np.arange(0,2240,step=80)#.reshape((-1, 1))
iodine = [13753, 10426, 8268, 7416, 6557, 5745, 5257, 4690, 4472, 4198, 3898, 3758, 3555, 3463, 
          3276, 3154, 3013, 2978, 2819, 2796, 2638, 2538, 2407, 2484, 2361, 2323, 2311, 2175]

tsqrt=np.sqrt(t)

time = np.log(t[1:])
i = iodine[1:]

a1=np.ones((len(t),2))
for a in range(len(a1)):
    a1[a][0]=tsqrt[a]

b1 = np.log(iodine)

solution=np.linalg.lstsq(a1,b1)
sol=solution[0]
res=solution[1]
sol_line=sol[0]*t+sol[1]


f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.scatter(t,iodine,label='Data')
ax1.plot(t, np.exp(sol[1])*np.exp(sol[0]*np.sqrt(t)),color = 'r',label="Model")
ax1.legend()
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Iodine")
ax1.set_title("Linearized model")

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.scatter(t,np.log(iodine))
ax2.set_xlabel("Time")
ax2.set_ylabel("ln(iodine)")
ax2.set_title("Semi-log plot")


def getSol(y, x, tol):
    a1=np.ones((len(x),2))
    for a in range(len(a1)):
        a1[a][0]=np.sqrt(x[a])
        
    c=np.e
    b=np.log(y-c)
    solution=np.linalg.lstsq(a1,b)
    sol=solution[0]
    #res=solution[1]
    sol_line=(sol[0]*np.sqrt(x)+sol[1])
    res=np.sum(np.abs(iodine-(np.exp(sol_line)+c))**2)
    
    win=1
    while win>tol:
        c1=c+win
        b1=np.log(y-c1)
        solution1=np.linalg.lstsq(a1,b1)
        sol1=solution1[0]
        #res1=solution1[1]
        sol_line1=(sol1[0]*np.sqrt(x)+sol1[1])
        res1=np.sum(np.abs(iodine-(np.exp(sol_line1)+c1))**2)
        
        c2=c-win
        b2=np.log(y-c2)
        solution2=np.linalg.lstsq(a1,b2)
        sol2=solution2[0]
        #res2=solution2[1]
        sol_line2=(sol2[0]*np.sqrt(x)+sol2[1])
        res2=np.sum(np.abs(iodine-(np.exp(sol_line2)+c2))**2)
        
        
        #print(res,res1,res2)
        if res1<res2 and res1<res:
            c=c1
            res=res1
            solution=solution1
        elif res2<res1 and res2<res:
            c=c2
            res=res2
            solution=solution2
        elif res<res1 and res<res2:
            win=win/2
            
        
            
    return solution, c

iodine = np.asarray(iodine)            

solution2, c = getSol(iodine, t, .01)
sol2 =  solution2[0]
sol_line2 = (sol2[0]*np.sqrt(t)+sol2[1])


f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.scatter(t, iodine,label='Data')
ax3.plot(t, np.exp(sol_line2)+c,c='r',label='Model')
ax3.set_title("Linearized model with additive constant")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Iodine")
ax3.legend()

r = r2_score(iodine, np.exp(sol[1])*np.exp(sol[0]*np.sqrt(t)))
r2 = r2_score(iodine,np.exp(sol_line2)+c)
