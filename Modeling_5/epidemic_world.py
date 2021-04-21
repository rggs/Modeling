#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:58:56 2020

@author: ryanswope
"""

from actor import World, Actor
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, *params):
    a,b,c,d=params
    return a*np.exp(-(b*x)+c) + d
    #return a*(1/(x-b))+c

def logistic_fit(t,*args):
    N,k,P0=args
    A=1-P0/N
    
    P=(A *np.e**(-k*t) - N)
    return P


##################### Two Guys####################
EpWorld=World(20)

spot=EpWorld.findSpot(1)
critter1=Actor(EpWorld, 1, spot[0], spot[1] )
spot=EpWorld.findSpot(2)
critter2=Actor(EpWorld, 2, spot[0], spot[1] )

critters=[critter1,critter2]
#EpWorld.place(critter.xpos, critter.ypos, critter.state)

critter1x=[critter1.xpos]
critter1y=[critter1.ypos]

critter2x=[critter2.xpos]
critter2y=[critter2.ypos]

allsick=False

while not allsick:
    for guy in critters:
        guy.getSick()
    for guy in critters:
        guy.move()
    critter1x.append(critter1.xpos)
    critter1y.append(critter1.ypos)
    
    critter2x.append(critter2.xpos)
    critter2y.append(critter2.ypos)
    
    if 1 in EpWorld.grid:
        allsick=False
    else:
        allsick=True
    #EpWorld.place(critter.xpos,critter.ypos,critter.state)
    




f3=plt.figure()
ax3=f3.add_subplot(111)
ax3.plot(critter1x, critter1y,color='blue',linewidth=0.5)
ax3.plot(critter2x, critter2y,color='red',linewidth=0.5)
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
ax3.set_title('Random Walk of Two People, ' + str(len(critter1x)-1)+ ' Steps')

####################### Many Guys #########################
popdense=[]
popsteps=[]
A=100
def count(grid, val):
    count=0
    for i in grid:
        for j in i:
            if j==val:
                count+=1
    return count

for N in range(100,A**2,100):

    EpWorld=World(A)
    
    #N=4000
    
    guy_list=[]
    
    #Health Pop
    for i in range(0,N):
        spot=EpWorld.findSpot(1)
        guy_list.append(Actor(EpWorld, 1, spot[0], spot[1] ))
        
    #Sick Pop
    spot=EpWorld.findSpot(2)
    guy_list.append(Actor(EpWorld, 2, spot[0], spot[1] ))
    
    #Simulate
    steps=[0]
    allsick=False
    numsick=[1]
    
    while not allsick:
        for guy in guy_list:
            guy.getSick()
        for guy in guy_list:
            guy.move()
        
        if 1 in EpWorld.grid:
            allsick=False
        else:
            allsick=True
        counter=count(EpWorld.grid,2)
        numsick.append(counter)
        steps.append(steps[-1]+1)
        # print('Steps='+ str(steps[-1]))
        # print('Infected='+ str(numsick[-1]))
        print(str((N+1)/A**2) +', '+str(steps[-1]))
        
        
    popdense.append((N+1)/A**2)
    popsteps.append(steps[-1])
    
# f4=plt.figure()
# ax4=f4.add_subplot(111)
# ax4.plot(steps, numsick, label='Epidemic Spread')
# ax4.set_xlabel('Steps')
# ax4.set_ylabel('Sick Population')
# ax4.set_title('Epidemic Spread, Sigma='+str((N+1)/A**2))
    
popt, pcov = curve_fit(func, np.asarray(popdense), np.asarray(popsteps), p0=[1,1,2,1])
fit=func(np.asarray(popdense),*popt)
'''
popt, pcov = curve_fit(logistic_fit, np.asarray(popdense), np.asarray(popsteps), p0=[1,1,3,])
fit2=logistic_fit(np.asarray(popdense),*popt)
'''
    
f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.plot(popdense, popsteps, 'b.', label='Simulation Data')
ax5.set_xlabel('Population Density')
ax5.set_ylabel('Number of Steps to Infect the Entire Population')
ax5.set_title('Disease Spread Speed by Population Density')
ax5.plot(popdense, fit, color='red', label='Inverse Fit')
#ax5.plot(popdense, fit2, color='green', label='Logistic Fit')
ax5.legend()

