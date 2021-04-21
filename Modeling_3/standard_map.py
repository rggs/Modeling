#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:31:11 2020

@author: ryanswope
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

p_ini=np.linspace(0,2*np.pi,20)
t_ini=np.linspace(0,2*np.pi,20)
k=.5

def next_step(t,p,L):
    #p=((p-k*np.sin(t))%(2*np.pi))
    #Zaslavsky p:
    p=(np.e**-L)*((p-k*np.sin(t))%(2*np.pi))
    t=(t+p)%(2*np.pi)
    return (t,p)
#k=np.linspace(.01,2*np.pi,50,endpoint=True)
L=np.linspace(0,2,50,endpoint=True)
z=10
for L in L:
    results=[]
    for p in p_ini:
        for t in t_ini:
            conds=[[t,p]]
            new_t=t
            new_p=p
            for i in range(150):
                new_t,new_p=next_step(new_t,new_p,L)
                
                conds.append([new_t,new_p])
            results.append(conds)
            
    f1=plt.figure()
    ax1=f1.add_subplot(111)
    for i in results:
        x=[]
        y=[]
        for j in i:
            x.append(j[0])
            y.append([j[1]])
        ax1.scatter(x,y,s=0.01)
        ax1.set_xlabel('Phase Angle')
        ax1.set_ylabel('Phase Momentum')
        ax1.set_title('Zaslavski Map with Attenuation='+str(L))
    filename='/home/users/rswope/Modeling/Modeling3/Zaslavski/zas_neg'+str(z)+'.png'
    z+=1
    plt.savefig(filename,dpi=1200)
