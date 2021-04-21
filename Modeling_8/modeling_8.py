#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:19:12 2020

@author: ryanswope
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


t=np.arange(0,2240,step=80)
iodine=np.asarray([13753, 10426, 8268, 7416, 6557, 5745, 5257, 4690, 4472, 4198, 3898, 3758, 3555, 3463, 3276, 3154, 3013, 2978, 2819, 2796, 2638, 2538, 2407, 2484, 2361, 2323, 2311, 2175])
#yint=np.ones(t.shape)
tsqrt=np.sqrt(t)

a1=np.ones((len(t),2))
for a in range(len(a1)):
    a1[a][0]=tsqrt[a]

b1=np.log(iodine)


solution=np.linalg.lstsq(a1,b1)
sol=solution[0]
sol_line=sol[0]*np.sqrt(t)+sol[1]
res=np.sum(np.abs(iodine-(np.exp(sol_line)))**2)/(len(t)-1)
iod_pred=np.exp(sol[1])*np.exp(sol[0]*np.sqrt(t))
residual1=np.sum(iodine-iod_pred)
r21=sklearn.metrics.r2_score(iodine, iod_pred )


'''
# f1=plt.figure()
# ax1=f1.add_subplot(111)
# ax1.plot(t, np.log(iodine),label='Observed')
# ax1.plot(t, sol_line ,label='Model')
# ax1.set_xlabel('t')
# ax1.set_ylabel(('ln(Iodine)'))
# ax1.legend()
'''

f2=plt.figure()
ax2=f2.add_subplot(111)
ax2.plot(t, iodine, 'r.', label='Data')
ax2.plot(t, iod_pred, label='Model')
ax2.set_title('No Additive Constant, R^2='+str(r21))
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Iodine Concentration')




def getSol(y, x, tol):
    a1=np.ones((len(x),2))
    for a in range(len(a1)):
        a1[a][0]=np.sqrt(x[a])
        
    s=len(x)-1
    c=np.e
    b=np.log(y-c)
    solution=np.linalg.lstsq(a1,b)
    sol=solution[0]
    sol_line=(sol[0]*np.sqrt(x)+sol[1])
    res=np.sum(np.abs(iodine-(np.exp(sol_line)+c))**2)/s
    
    win=1
    while win>tol:
        c1=c+win
        b1=np.log(y-c1)
        solution1=np.linalg.lstsq(a1,b1)
        sol1=solution1[0]
        #res1=solution1[1]
        sol_line1=(sol1[0]*np.sqrt(x)+sol1[1])
        res1=np.sum(np.abs(iodine-(np.exp(sol_line1)+c1))**2)/s
        
        c2=c-win
        b2=np.log(y-c2)
        solution2=np.linalg.lstsq(a1,b2)
        sol2=solution2[0]
        #res2=solution2[1]
        sol_line2=(sol2[0]*np.sqrt(x)+sol2[1])
        res2=np.sum(np.abs(iodine-(np.exp(sol_line2)+c2))**2)/s
        
        
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
    #print(res) 
        
            
    return solution, c
            
    

solution2, c=getSol(iodine, t, .001)
sol2=solution2[0]
sol_line2=(sol2[0]*np.sqrt(t)+sol2[1])

iod_pred2=np.exp(sol_line2)+c
residual2=np.sum(iodine-iod_pred2)
r22=sklearn.metrics.r2_score(iodine, iod_pred2 )
print('Residuals '+str(round(residual2,4)))

'''
# f4=plt.figure()
# ax4=f4.add_subplot(111)
# ax4.plot(t, np.log(iodine),label='Observed')
# ax4.plot(t, sol_line2 ,label='Model')
# ax4.set_xlabel('t')
# ax4.set_ylabel(('ln(Iodine)'))
# ax4.legend()
'''



f3=plt.figure()
ax3=f3.add_subplot(111)
ax3.plot(t, iodine,'r.', label='Data')
ax3.plot(t, iod_pred2,label='Model')
ax3.legend()
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Iodine Concentration')
ax3.set_title('Additive Constant, R^2='+str(round(r22,4)))



# s**2=res/n-2

print('Part 3')
    
####################### Part 3 #########################
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
import astropy

def P(n, x):  
    if(n == 0): 
        return 1 # P0 = 1 
    elif(n == 1): 
        return x # P1 = x 
    else: 
        return (((2 * n)-1)*x * P(n-1, x)-(n-1)*P(n-2, x))/float(n)
    
    
def sigmaClip(xarr, yobs, ymodel, upperlimit, lowerlimit):
    newx=[]
    newy=[]
    
    n=len(xarr)
    res=np.sum(yobs-ymodel)
    sigma=np.abs(res/(n-1))
    uplim=upperlimit*sigma
    lowlim=sigma*lowerlimit

    print(sigma)
    
    for i in range(len(xarr)):
        if (yobs[i] < ymodel[i]+uplim) and (yobs[i]>ymodel[i]-lowlim):
            newx.append(xarr[i])
            newy.append(yobs[i])
    return newx, newy
          

def legFit(xdata, ydata, order):
    arr=[]
    for i in xdata:
        row=np.zeros(order)
        for j in range(order):
            row[j]=P(j, i)
        arr.append(row)
    
    arr=np.asarray(arr)
    c_arr=np.linalg.lstsq(arr, ydata)[0]
    
    ymodel=np.dot(arr, c_arr)
    
    n=len(xdata)
    res=np.sum(ydata-ymodel)
    sigma=np.abs(res/(n-1))
    return ymodel, sigma
    
#'''
# def chunkClip(xarr, yarr, upperlimit, lowerlimit, chunk_length, order):
#     newx=[]
#     newy=[]
#     num_chunks=len(xarr)//chunk_length
#     remainder=len(xarr)%chunk_length
#     for i in range(num_chunks):
#         if i<num_chunks:
#             x=xarr[i*chunk_length:(i+1)*chunk_length]
#             y=yarr[i*chunk_length:(i+1)*chunk_length]
#             ymodel, sig =legFit(x,y,order)
#             xx,yy=sigmaClip(x,y,ymodel,upperlimit, lowerlimit)
#             for j in range(len(xx)):
#                 newx.append(xx[j])
#                 newy.append(yy[j])
#         else:
#             x=xarr[i*chunk_length:]
#             y=yarr[i*chunk_length:]
#             ymodel, sig=legFit(x,y,order)
#             xx,yy=sigmaClip(x,y,ymodel,upperlimit, lowerlimit)
#             for j in range(len(xx)):
#                 newx.append(xx[j])
#                 newy.append(yy[j])
#     return newx, newy
    #'''
    



    
data=np.genfromtxt('/Users/ryanswope/PythonDocs/vega.txt', delimiter=' ')
wave=data[:,0]
flux=data[:,3]
max_wav=np.max(wave)
wave=wave/max_wav

#First Fit
arr=[]
order=5
for i in wave:
    row=np.zeros(order)
    for j in range(order):
        row[j]=P(j, i)
    arr.append(row)

arr=np.asarray(arr)
c_arr=np.linalg.lstsq(arr, flux)[0]

#newx, newy = sigmaClip(wave, flux, np.dot(arr, c_arr), 5e-6, 1.5e-6)


##Savgol Filter
y_sav=savgol_filter(flux, 99, 7)
#Second Fit
ymodel, sig=legFit(wave, y_sav, 20)

newx, newy = sigmaClip(wave, y_sav, ymodel, 70, 45e2)

ymodel2, sig2=legFit(newx, newy, 15)

wave=np.array(wave)*max_wav
newx=np.array(newx)*max_wav

f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.plot(wave, flux, 'r.', label='Data')
# ax5.plot(wave, np.dot(arr, c_arr), label='Model')
# ax5.plot(wave, y_sav, 'b.')
ax5.plot(newx, newy, 'k.', label='Sigma Clipped Data')
# ax5.plot(wave, ymodel,label='25')
ax5.set_xlabel('Wavelength (A)')
ax5.set_ylabel('Flux')
ax5.set_title('Absorption Spectrum of Vega')
ax5.legend()


data2=astropy.io.ascii.read('/Users/ryanswope/Desktop/tau.txt')
l_wave=list(data2['col1'])
l_flux=list(data2['col2'])
new_wave=[]
new_flux=[]

for i in range(len(l_wave)):
    if l_flux[i]>0:
        new_wave.append(l_wave[i])
        new_flux.append(l_flux[i])
l_wave=np.array(new_wave)
l_flux=np.array(new_flux)
max_val=np.max(l_wave)
l_wave=l_wave/max_val


l_y_sav=savgol_filter(l_flux, 99, 7)
#Second Fit
l_ymodel, l_sig=legFit(l_wave, l_y_sav, 20)

l_newx, l_newy = sigmaClip(l_wave, l_y_sav, l_ymodel, 7000, 45e6)

l_ymodel2, l_sig2=legFit(l_newx, l_newy, 15)

renorm=l_wave*max_val
l_newx=np.array(l_newx)*max_val

f6=plt.figure()
ax6=f6.add_subplot(111)
ax6.plot(renorm, l_flux, 'r.', label='Data')
# ax6.plot(renorm, l_ymodel, label='Model')
# ax6.plot(renorm, l_y_sav, 'b.')
ax6.plot(l_newx, l_newy, 'k.', label='Sigma Clipped Data')
# ax6.plot(l_newx, l_ymodel2,label='25')
ax6.set_xlabel('Wavelength (micrometers)')
ax6.set_ylabel('Flux (W/m^2/um^-1)')
ax6.set_title('Absorption Spectrum Of h Tau')
ax6.set_xlim(np.min(renorm), 1.2)
ax6.legend()