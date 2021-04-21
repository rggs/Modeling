#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 02:02:13 2020

@author: ryanswope
"""
import numpy as np
import scipy as scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq


P=1
w=2*np.pi*P
G=6.67e-11*((9.95839577e-14)**(-2))*2.98692e-34
Msun=1.989e30
Mjup=1.898e27
Me=5.972e24
q=Me/Msun
#q=1
a=1
w=2*np.pi
T=1000*np.pi


y0=[0,0,1,0]
cmx1=-(q*a)/(1+q)
cmx2=a+cmx1
y0_cm=[y0[0]+cmx1,0,cmx2,0]

def x_accel(x, ydot):
    s1=np.abs(x-y0[0])
    s2=np.abs(x-y0[2])
    delx=(-2/(1+q))*((x-y0[0])/s1**3) + ((-2*q)/(1+q))*((x-y0[2])/s2**3) + 2*x + 4*ydot
    return delx


def deriv(t, X):
    x1=y0_cm[0]
    x2=y0_cm[2]
    y1=y0_cm[1]
    y2=y0_cm[3]
    x, y, z, xdot, ydot, zdot = X
    s1=((x-x1)**2 + ((y-y1)**2) + z**2)**(1/2)
    s2=((x-x2)**2 + ((y-y2)**2) + z**2)**(1/2)
    xddot = x + 2*ydot  -  ((1-q)/s1**3)*(x+q) - (q/s2**3)*(x-(1-q))
    yddot = y - 2*xdot  -  ((1-q)/s1**3)*y      - (q/s2**3)*y
    zddot =             -  ((1-q)/s1**3)*z      - (q/s2**3)*z
    return (xdot, ydot, zdot, xddot, yddot, zddot)
    
    
#Change exponents in denominator of gravitational force to 3
#Plot contours of potential, not force
    
#initial conditions of the asteroid
    
ydot0s   = np.linspace(-0.08, 0.08, 20)
x0ydot0s = []
for ydot0 in ydot0s:
    x0, infob =  brentq(x_accel, -1.5, -0.5, args=(ydot0), xtol=1E-11, rtol=1E-11,
                           maxiter=100, full_output=True, disp=True)
    x0ydot0s.append((x0, ydot0))
    
states = [np.array([x0, 0, 1e-9, 0, ydot0, 0]) for (x0, ydot0) in x0ydot0s]

P2=[(q+1),a*(2*q+3),(a**2)*(q+3),-q*(a**3),-2*(a**4)*q,-(a**5)*q]


L4=[a*((1/2)*((1-q)/(1+q))), a*np.sqrt(3)/2]
L2=[cmx2+np.roots(P2)[2],0]
y1=[L4[0],L4[1],1e-4,0,.0001,0]
#y1=states[8]
sol=integrate.RK45(fun=deriv,t0=0,y0=y1,t_bound=(T), first_step=.1, max_step=.1,rtol=1e-8, atol=0)
#sol = integrate.odeint(diff_eq, y1, times, atol = 1E-11, full_output=True)

T2=int(T/.1)+1
x_vals=[]
y_vals=[]
z_vals=[]

#y1=states[11]
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
ax7.set_title('Attempted JWST Orbit, q='+str(round(q,5)))
ax7.set_xlabel('X Position')
ax7.set_ylabel('Y Position')
ax7.scatter(y0[0],y0[1])
ax7.scatter(y0[2],y0[3])
#filename='/Users/ryanswope/PythonDocs/Modeling_2/Stability_Gif/stability'+str(j)+'.png'
#filename='~/Modeling/Stability_Gif/stability'+str(j)+'.png'
#plt.savefig(filename,dpi=300)
#plt.gca()
