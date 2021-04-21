#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:32:10 2020

@author: ryanswope
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


'''

#Define Iterative Function

def logistic(r,iters, xn,xn1,xn2):
    xn_new=r*xn*(1-xn)
    if(iters==0):
        return (r,xn_new)
    elif(xn_new==xn1 or xn_new==xn2):
        return (r,xn_new)
    else:
        return logistic(r,iters-1,xn_new,xn,xn1)
'''

def logistic(r,iters, xn,xn1,xn2):
    i=0
    while i<iters:
        xn_new=r*xn*(1-xn)
        xn=xn_new
        i+=1
    return r, xn
#Define r's, iterations, initial x's
    
xs=np.linspace(0.01,.99,99)
rs=np.linspace(0,4,101)
iterations=np.linspace(200,210,6)

results=[]
for i in rs:
    for j in xs:
        for k in iterations:
            results.append(logistic(i,k,j,np.nan,np.nan))
            print(i,j)
            
r_results=[]
x_results=[]
unique=[]

for g in results:
    r_results.append(g[0])
    x_results.append(g[1])
    is_unique=True
    for t in unique:
        if (t[0]==g[0] and abs(t[1]-g[1])<1e-2):
            is_unique=False
    if(is_unique): unique.append(g)
    
unique_count=[[unique[0][0],1]]
for k in rs:
    counter=0
    for j in unique:
        if j[0]==k:
            counter+=1
    unique_count.append([k,counter])
    
bifurs=[]

for k in range(1,len(unique_count)):
    if(unique_count[k][1]>unique_count[k-1][1]):
        bifurs.append(unique_count[k][0])
        
smallx1=[]
smallr1=[]

smallx2=[]
smallr2=[]


for i in results:
    if bifurs[0]-.04<i[0]<bifurs[2]+.04:
        smallr1.append(i[0])
        smallx1.append(i[1])

    if bifurs[1]-.04<i[0]<bifurs[3]+.04 and i[1]>2/3:
        smallr2.append(i[0])
        smallx2.append(i[1])
        

            
f1=plt.figure()
ax1=f1.add_subplot(111)
ax1.set_xlabel('r')
ax1.set_ylabel('x')
ax1.set_title('Logistic Mapping')
ax1.scatter(r_results,x_results,s=.001,color='black')
ax1.vlines(bifurs[0:4],0,.2,linewidth=0.5)

f2=plt.figure()
ax2=f2.add_subplot(211)
ax2.set_xlabel('r')
ax2.set_title('First Bifurcation')
ax2.scatter(smallr1,smallx1,s=.1,color='red')

ax3=f2.add_subplot(212)
ax3.set_xlabel('r')
ax3.set_ylabel('x')
ax3.set_title('Second Bifurcation')
ax3.scatter(smallr2,smallx2,s=.1,color='blue')


#filename='/home/users/rswope/Modeling/Modeling3/logistic.png'
#plt.savefig(filename,dpi=1200)
