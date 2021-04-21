#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:00:38 2020

@author: ryanswope
"""

import numpy as np
import scipy as scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from scipy import integrate

#pd = .587
ecc=.967
#T=76*365
T=2*np.pi
tp=0
tolerance=1e-8
#a=pd/(1-ecc)
#G=6.67e-11*2.98692e-34*7464960000
#G = 7.37682304563492e-28
#Mass=1.989e30

T=T*5
Mass=1
G=1
a=1
pd=a*(1-ecc)
T1=int(T)




def M(t):
    return 2*np.pi*(((t-tp)%(2*np.pi))/(2*np.pi))

def f(E,t):
    return E-ecc*np.sin(E)-M(t)

def E(E0,t,tol):
    En=E0-(f(E0,t)/(1-(ecc*np.cos(E0))))
    if abs(f(En,t))<tol:
        return En
    else: return E(En,t,tol)
    
def theta(E):
    return 2*np.arctan((((1+ecc)/(1-ecc))**.5)*np.tan(E/2)) #Andrej mixed up + and - sign

def radius(theta):
    return a*(1-(ecc**2))/(1+ecc*np.cos(theta))

time = np.linspace(0,(T),num=((T*1000)+1))
r=[]
E_arr=[]
M_arr=[]
theta_arr=[]
for i in time:
    Mi=M(i)
    M_arr.append(Mi)
    Ei=E(M(i),i,tolerance)
    E_arr.append(Ei)
    thetai=theta(Ei)
    theta_arr.append(thetai)
    ri=radius(thetai)
    r.append(ri)
    
newton_peris=[]
for i in range(T1):
    if i%T==0:
        newton_peris.append(r[i])

newton_error=[]
for j in newton_peris:    
    newton_error.append(j-pd)


f1=plt.figure()
ax1=f1.add_subplot(111,projection='polar')
ax1.plot(theta_arr,r,'b--')
ax1.set_title("Orbit of Halley's Comet using True Anamoly and AU")
f1.tight_layout()


def diff_eq(t, pos):
    
    x,y,vx,vy=pos
    
    fdot=[vx, vy, -(G*Mass)/((x**2 + y**2)**(3/2))*x, -(G*Mass)/((x**2 + y**2)**(3/2))*y]
    return fdot


#y0=[p_d,0,0,0.0317652]
y0=[pd,0.,0.,np.sqrt(G*Mass*((2/pd)-(1/a)))]
#sol=integrate.RK45(fun=diff_eq,t0=0,y0=[p_d,0.,0.,0.0317652],t_bound=27740, first_step=1., max_step=1.,rtol=1e-8, atol=0)
#vel=integrate.solve_ivp(diff_eq,t_span=[0,27740],y0=[p_d,0,0,0.0317652],method='RK45',t_eval=time)
#vel=integrate.odeint(diff_eq, y0=y0, t=time,tfirst=True)

sol=integrate.RK45(fun=diff_eq,t0=0,y0=y0,t_bound=(T), first_step=.001, max_step=.001,rtol=1e-8, atol=0)
RK23sol=integrate.RK23(fun=diff_eq,t0=0,y0=y0,t_bound=(T), first_step=.001, max_step=.001,rtol=1e-8, atol=0)
LSODAsol=integrate.LSODA(fun=diff_eq,t0=0,y0=y0,t_bound=(T), first_step=.001, max_step=.001,rtol=1e-8, atol=1e-8)

x_vals=[]
y_vals=[]
runga_peris=[]

rk23x=[]
rk23y=[]

lsodax=[]
lsoday=[]

T2=int(T/.001)+1

for i in range(T2):
    values=LSODAsol.step()
    lsodax.append(LSODAsol.y[0])
    lsoday.append(LSODAsol.y[1])
    print(T2-i)
    
for i in range(T2):
    values=sol.step()
    x_vals.append(sol.y[0])
    y_vals.append(sol.y[1])
    if i%T==0:
        runga_peris.append(sol.y[0])
    print(T2-i)

for i in range(T2):
    values=RK23sol.step()
    rk23x.append(RK23sol.y[0])
    rk23y.append(RK23sol.y[1])
    print(T2-i)
    



newton_peris=[]
for i in range(T2):
    if i%T==0:
        newton_peris.append(r[i])

newton_error=[]
for j in newton_peris:    
    newton_error.append(j-pd)
    

runga_error=[]
for j in runga_peris:    
    runga_error.append(j-pd)
    
runga_radii=[]
for j in range(T2):
        radius=(x_vals[j]**2 + y_vals[j]**2)**(1/2)
        runga_radii.append(radius)
       
radius_err=[]
for j in range(T2):
        rad=abs(r[j]-runga_radii[j])
        radius_err.append(rad)

cum=np.cumsum(radius_err)
f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.plot(cum,label='RK45 Error')
ax5.legend()
ax5.set_title('Cumulative Error')
ax5.set_xlabel('Days')
ax5.set_ylabel('Total Error (AU)')



f3=plt.figure()
ax3=f3.add_subplot(111)
ax3.plot(runga_error,label='Runga')
ax3.plot(newton_error, label='Newton')
ax3.legend()

f4=plt.figure()
ax4=f4.add_subplot(111)
ax4.plot(radius_err,label='RK45')
ax4.legend()
ax4.set_title('Radii Errors')
ax4.set_xlabel('Days')
ax4.set_ylabel('Difference in Radii (AU)')

f6=plt.figure()
ax6=f6.add_subplot(111)
ax6.plot(r,label='Newton Method')
ax6.plot(runga_radii, label='RK45')
ax6.legend()
ax6.set_title('Stability between Newton and RK45')
ax6.set_xlabel('Days')
ax6.set_ylabel('Radius (AU)')

f7=plt.figure()
ax7=f7.add_subplot(111)
ax7.plot(x_vals, y_vals, label='RK45')
ax7.plot(rk23x,rk23y,label='RK23')
ax7.plot(lsodax,lsoday,label='LSODA')
ax7.legend()
ax7.set_title('Orbit With Various Methods')
ax7.set_xlabel('X Position (AU)')
ax7.set_ylabel('Y Position (AU)')



