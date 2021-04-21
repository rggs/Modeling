#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 02:39:04 2020

@author: ryanswope
"""

import numpy as np
import scipy as scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from scipy import integrate
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

G=6.67e-11*((9.95839577e-14)**(-2))*2.98692e-34
Msun=1.989e30
Mjup=1.898e27
q=Mjup/Msun
a=5.2
T=4300
w=2*np.pi/T

y0=[0,0,5.2,0]
def diff_eq(t, pos):
    
    x,y,vx,vy=pos
    
    fdot=[vx, vy, -(G*Msun)/((x**2 + y**2)**(3/2))*x + -(G*Mjup)/((x**2 + y**2)**(3/2))*x -x*(w**2)/2 + 2*w*vy, -(G*Msun)/((x**2 + y**2)**(3/2))*y + -(G*Mjup)/((x**2 + y**2)**(3/2))*y -y*(w**2)/2 + 2*w*vx]
    return fdot

#+ -(G*Mjup)/((x**2 + y**2)**(3/2))*x +x*(w**2)/2 + 2*w*vy
#+ -(G*Mjup)/((x**2 + y**2)**(3/2))*y +y*(w**2)/2 + 2*w*vx

#y0=[p_d,0,0,0.0317652]
L4=[a*((1/2)*((1-q)/(1+q))), a*np.sqrt(3)/2]
y1=[L4[0],L4[1],-400000,0]

#sol=integrate.RK45(fun=diff_eq,t0=0,y0=[p_d,0.,0.,0.0317652],t_bound=27740, first_step=1., max_step=1.,rtol=1e-8, atol=0)
#vel=integrate.solve_ivp(diff_eq,t_span=[0,27740],y0=[p_d,0,0,0.0317652],method='RK45',t_eval=time)
#vel=integrate.odeint(diff_eq, y0=y0, t=time,tfirst=True)

sol=integrate.RK45(fun=diff_eq,t0=0,y0=y1,t_bound=(T), first_step=.1, max_step=.01,rtol=1e-8, atol=0)

T2=int(T/.1)+1
x_vals=[]
y_vals=[]
z_vals=[]

for i in range(T2):
    values=sol.step()
    x_vals.append(sol.y[0])
    y_vals.append(sol.y[1])
    z_vals.append(sol.y[2])
    print(T2-i)
    
f7=plt.figure()
ax7=f7.add_subplot(111)
ax7.plot(x_vals, y_vals, label='RK45')
ax7.legend()
ax7.set_title('Trojan Orbit')
ax7.set_xlabel('X Position')
ax7.set_ylabel('Y Position')
ax7.scatter(y0[0],y0[1])
ax7.scatter(y0[2],y0[3])