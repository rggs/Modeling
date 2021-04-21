#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:27:58 2020

@author: ryanswope
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.io import wavfile

# x1 = np.linspace(-2*np.pi, 2*np.pi, 500, endpoint=True)

# sine=np.sin(x1)
# cosine=np.cos(x1)

# x2=np.linspace(-1.5,1.5,301, endpoint=True)

# sq_pulse=[]
# for i in x2:
#     if i<-.5:
#         sq_pulse.append(0)
#     elif -.5 <= i < .5:
#         sq_pulse.append(1)
#     else:
#         sq_pulse.append(0)
        
# tri=[]
# for i in x2:
#     if i<-1:
#         tri.append(0)
#     elif -1 <= i <0:
#         tri.append(i+1)
#     elif 0 <= i < 1:
#         tri.append(-i+1)
#     else:
#         tri.append(0)
        
# delta=[]
# for i in x2:
#     if i==0:
#         delta.append(1)
#     else:
#         delta.append(0)
        
# f1=plt.figure()
# ax1=f1.add_subplot(231)
# ax1.plot(x1, sine)
# ax1.set_title('Sine')
# ax2=f1.add_subplot(232)
# ax2.plot(x1,cosine)
# ax2.set_title('Cosine')
# ax3=f1.add_subplot(233)
# ax3.plot(x2, sq_pulse)
# ax3.set_title('Square Pulse')
# ax4=f1.add_subplot(234)
# ax4.plot(x2, tri)
# ax4.set_title('Trianlge Pulse')
# ax5=f1.add_subplot(235)
# ax5.plot(x2, delta)
# ax5.set_title('Delta')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Amplitude')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Amplitude')
# ax3.set_xlabel('Time')
# ax3.set_ylabel('Amplitude')
# ax4.set_xlabel('Time')
# ax4.set_ylabel('Amplitude')
# ax5.set_xlabel('Time')
# ax5.set_ylabel('Amplitude')
# f1.tight_layout()

# f_sin=np.fft.fft(sine)
# f_cos=np.fft.fft(cosine)
# f_sq=np.fft.fft(sq_pulse)
# f_tri=np.fft.fft(tri)
# f_delta=np.fft.fft(delta)

# if_sin=np.fft.ifft(sine)
# if_cos=np.fft.ifft(cosine)
# if_sq=np.fft.ifft(sq_pulse)
# if_tri=np.fft.ifft(tri)
# if_delta=np.fft.ifft(delta)

# fr1=np.fft.fftfreq(x1.shape[-1])
# fr2=np.fft.fftfreq(x2.shape[-1])

# f2=plt.figure()
# #f2.suptitle('Fourier Transformed')
# ax1=f2.add_subplot(231)
# ax1.plot(fr1, f_sin)
# ax1.set_title('Sine')
# ax2=f2.add_subplot(232)
# ax2.plot(fr1,f_cos)
# ax2.set_title('Cosine')
# ax3=f2.add_subplot(233)
# ax3.plot(fr2, f_sq)
# ax3.set_title('Square Pulse')
# ax4=f2.add_subplot(234)
# ax4.plot(fr2, f_tri)
# ax4.set_title('Trianlge Pulse')
# ax5=f2.add_subplot(235)
# ax5.plot(fr2, f_delta)
# ax5.set_title('Delta')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Amplitude')
# ax3.set_xlabel('Frequency')
# ax3.set_ylabel('Amplitude')
# ax4.set_xlabel('Frequency')
# ax4.set_ylabel('Amplitude')
# ax5.set_xlabel('Frequency')
# ax5.set_ylabel('Amplitude')
# f2.tight_layout()

# f3=plt.figure()
# #f3.suptitle('Inverse Fourier Transformed')
# ax1=f3.add_subplot(231)
# ax1.plot(fr1, if_sin)
# ax1.set_title('Sine')
# ax2=f3.add_subplot(232)
# ax2.plot(fr1,if_cos)
# ax2.set_title('Cosine')
# ax3=f3.add_subplot(233)
# ax3.plot(fr2, if_sq)
# ax3.set_title('Square Pulse')
# ax4=f3.add_subplot(234)
# ax4.plot(fr2, if_tri)
# ax4.set_title('Trianlge Pulse')
# ax5=f3.add_subplot(235)
# ax5.plot(fr2, if_delta)
# ax5.set_title('Delta')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Amplitude')
# ax3.set_xlabel('Frequency')
# ax3.set_ylabel('Amplitude')
# ax4.set_xlabel('Frequency')
# ax4.set_ylabel('Amplitude')
# ax5.set_xlabel('Frequency')
# ax5.set_ylabel('Amplitude')
# f3.tight_layout()

# #Power Spectral Densities
# fx1=(x1[-1]-x1[0])/len(x1)
# fx2=(x1[-1]-x2[0])/len(x2)

# sin_psd=periodogram(sine)
# cos_psd=periodogram(cosine)
# sq_psd=periodogram(sq_pulse)
# tri_psd=periodogram(tri)
# delta_psd=periodogram(delta)

# f4=plt.figure()
# #f3.suptitle('Inverse Fourier Transformed')
# ax1=f4.add_subplot(231)
# ax1.plot(sin_psd[0], sin_psd[1])
# ax1.set_title('Sine')
# ax2=f4.add_subplot(232)
# ax2.plot(cos_psd[0],cos_psd[1])
# ax2.set_title('Cosine')
# ax3=f4.add_subplot(233)
# ax3.plot(sq_psd[0], sq_psd[1])
# ax3.set_title('Square Pulse')
# ax4=f4.add_subplot(234)
# ax4.plot(tri_psd[0], tri_psd[1])
# ax4.set_title('Trianlge Pulse')
# ax5=f4.add_subplot(235)
# ax5.plot(delta_psd[0], delta_psd[1])
# ax5.set_title('Delta')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Power')
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Power')
# ax3.set_xlabel('Frequency')
# ax3.set_ylabel('Power')
# ax4.set_xlabel('Frequency')
# ax4.set_ylabel('Power')
# ax5.set_xlabel('Frequency')
# ax5.set_ylabel('Power')
# f4.tight_layout()

# #Part b
# x=np.linspace(0,20*np.pi, 200000)
# x2=[]
# for i in range(len(x)):
#     if i%200==0:
#         x2.append(x[i])
# x2=np.array(x2)
# y=np.cos(x)+np.cos(x/3)
# y2=np.cos(x2)+np.cos(x2/3)
# ff=np.fft.fft(y)
# ff2=np.fft.fft(y2)
# ff_psd=periodogram(x)
# ff2_psd=periodogram(x2,fs=200)



# f1=np.fft.fftfreq(x.shape[-1])
# f2=np.fft.fftfreq(x2.shape[-1])

# f5=plt.figure()
# #f5.suptitle('Effects of Aliasing, Fourier Transform')
# ax1=f5.add_subplot(121)
# ax1.plot(f1, ff, 'k.')
# ax1.set_xlim(-.005,.005)
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Amplitude')
# ax2=f5.add_subplot(122)
# ax2.plot(f2,ff2, 'r.')
# ax2.set_xlim(-.005,.005)
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Amplitude')
# f5.tight_layout()

# f6=plt.figure()
# #f6.suptitle('Effects of Aliasing, PSD')
# ax1=f6.add_subplot(121)
# ax1.plot(ff_psd[0], ff_psd[1], 'k')
# ax1.set_xlabel('Frequency')
# ax1.set_ylabel('Power')
# ax1.set_yscale('log')

# ax2=f6.add_subplot(122)
# ax2.plot(ff2_psd[0],ff2_psd[1], 'r')
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Power')
# ax2.set_yscale('log')
# f6.tight_layout()

##Part C

# guitarsr, guitar=wavfile.read('/Users/ryanswope/Desktop/sp_data/guitar.wav')
# #guitar=guitar[:,0]
# x=np.linspace(0, len(guitar), len(guitar))
# x=x/guitarsr
# #sample_rate=len(x)/x[-1]
# f=np.fft.fftfreq(x.shape[-1])
# f=f*guitarsr
# f_p=np.fft.fft(guitar)



# f5=plt.figure()
# ax1=f5.add_subplot(211)
# ax1.plot(x, guitar)
# ax1.set_xlabel('Seconds')
# ax1.set_ylabel('Amplitude')
# ax2=f5.add_subplot(212)
# ax2.plot(f, f_p)
# ax2.set_xlabel('Hertz')
# ax2.set_ylabel('Amplitude')
# f5.tight_layout()

#Part d
# def tran_Bach(filename, sr):
#     bach=np.genfromtxt(filename)
#     x=np.linspace(0,len(bach), len(bach))
    
#     x=x/sr
#     f=np.fft.fft(bach)
#     fr=np.fft.fftfreq(x.shape[-1])
#     fr=fr*sr
#     return fr, f

# fr0, f0=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.1378.txt', 882)
# fr1, f1=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.1378.txt', 1378)
# fr2, f2=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.2756.txt', 2756)
# fr3, f3=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.5512.txt', 5512)
# fr4, f4=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.11025.txt', 11025)
# fr5, f5=tran_Bach('/Users/ryanswope/Desktop/sp_data/Bach.44100.txt', 44100)

# f6=plt.figure()
# ax0=f6.add_subplot(611)
# ax0.plot(fr0, f0)
# ax0.set_title('882')
# ax1=f6.add_subplot(612)
# ax1.plot(fr1, f1)
# ax1.set_title('1378')
# ax2=f6.add_subplot(613)
# ax2.plot(fr2, f2)
# ax2.set_title('2756')
# ax3=f6.add_subplot(614)
# ax3.plot(fr3, f3)
# ax3.set_title('5512')
# ax4=f6.add_subplot(615)
# ax4.plot(fr4, f4)
# ax4.set_title('11025')
# ax5=f6.add_subplot(616)
# ax5.plot(fr5, f5)
# ax5.set_title('44100')
# # ax1.set_xlabel('Frequency (Hz)')
# # ax1.set_ylabel('Amplitude')
# # ax0.set_xlabel('Frequency (Hz)')
# # ax0.set_ylabel('Amplitude')
# # ax2.set_xlabel('Frequency (Hz)')
# # ax2.set_ylabel('Amplitude')
# # ax3.set_xlabel('Frequency (Hz)')
# # ax3.set_ylabel('Amplitude')
# # ax4.set_xlabel('Frequency (Hz)')
# # ax4.set_ylabel('Amplitude')
# # ax5.set_xlabel('Frequency (Hz)')
# # ax5.set_ylabel('Amplitude')
# f6.tight_layout()

# #Part e
# dodge_cars=np.genfromtxt('/Users/ryanswope/Desktop/sp_data/dodgers.cars.data')
# dodge_events=np.genfromtxt('/Users/ryanswope/Desktop/sp_data/dodgers.events.data')
# f7=plt.figure()
# ax1=f7.add_subplot(111)
# ax1.plot(dodge_cars[:,0], dodge_cars[:,1], label='Car Density')
# ax1.set_ylabel('Car Density')
# ax2=ax1.twinx()
# ax2.plot(dodge_events[:,0], dodge_events[:,2], label='Start Time')
# ax2.plot(dodge_events[:,1], dodge_events[:,2], label='End Time')
# ax2.set_ylabel('Attendance')
# ax1.legend()
# ax1.set_title('Car Density and Attendance')


# #dodge_cars[:,0]=dodge_cars[:,0]-dodge_cars[:,0][0]
# dodge_cars[:,0]=dodge_cars[:,0]/(60*60)

# sr=len(dodge_cars[:,0])/(dodge_cars[:,0][-1]-dodge_cars[:,0][0])

# car_fr=np.fft.fftfreq(dodge_cars[:,0].shape[-1])*sr
# car_f=np.fft.fft(dodge_cars[:,1])
# f9=plt.figure()
# ax1=f9.add_subplot(111)
# ax1.plot(car_fr, car_f)
# ax1.set_ylabel('Amplitude')
# ax1.set_xlabel('Frequency')
# ax1.set_title('Fourier Transform of Car Density')
# f9.tight_layout()




def autocorrelate(x):
    auto=np.correlate(x,x,mode='full')
    return auto[auto.size // 2:]

# #Part f
# boil=np.genfromtxt('/Users/ryanswope/Desktop/sp_data/boiling.data')
# auto_b=autocorrelate(boil)

# f8=plt.figure()
# ax1=f8.add_subplot(111)
# ax1.plot(auto_b, 'b.')
# ax1.set_yscale('log')
# ax1.set_xlabel('Lag')
# ax1.set_ylabel('Autocorrelation')
# ax1.set_title('Autocorrelation of Boiling Water')

#Part g
sun_month=np.genfromtxt('/Users/ryanswope/Desktop/sp_data/sunspots.monthly.data')
sun_year=np.genfromtxt('/Users/ryanswope/Desktop/sp_data/sunspots.yearly.data')
sun_month_time=(sun_month[:,0]+((sun_month[:,1]-1)/12))*100
sun_month_data=sun_month[:,2]

sun_year_time=sun_year[:,0]
sun_year_data=sun_year[:,1]

sun_yf=np.fft.fft(sun_year_data)
sun_yfr=np.fft.fftfreq(sun_year_time.shape[-1])

sun_mf=np.fft.fft(sun_month_data)
sun_mfr=np.fft.fftfreq(sun_month_time.shape[-1])

f10=plt.figure()
ax1=f10.add_subplot(211)
ax1.plot(sun_yfr, sun_yf)
ax1.set_title('Yearly Sunspots')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Amplitude')
ax2=f10.add_subplot(212)
ax2.plot(sun_mfr, sun_mf)
ax2.set_title('Monthly Sunspots')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Amplitude')
f10.tight_layout()

month_auto=autocorrelate(sun_month_data)
year_auto=autocorrelate(sun_year_data)

f11=plt.figure()
ax1=f11.add_subplot(211)
ax1.plot(month_auto, 'k.')
ax1.set_yscale('log')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')
ax2=f11.add_subplot(212)
ax2.plot(year_auto, 'k.')
ax2.set_yscale('log')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')
ax1.set_title('Yearly Sunspots')
ax2.set_title('Monthly Sunspots')
f11.tight_layout()

# #Spectrogram
# spectrum=np.linspace(400,800,10000)
# x=np.linspace(0,8,10000)
# wavex=[]
# wavey=[]
# for i in spectrum:
#     wavex.append(i*x)
#     wavey.append(np.sin(2*np.pi*wavex[-1]/i))
    
# time_delay=2e-6
# wave_delay=np.array(wavex)-(2e-6*(3e8))
# wave_delay=list(wave_delay)
# wavedely=[]
# for i in range(len(spectrum)):
#     wavedely.append(np.sin(2*np.pi*wave_delay[i]/spectrum[i]))
    
# f12=plt.figure()
# ax1=f12.add_subplot(111)
# ax1.plot(wavex[0], wavey[0])
# ax1.plot(wave_delay[0], wavedely[0])

# conv_vals=[]
# for i in range(len(spectrum)):
#     conv_vals.append(np.convolve(wavey[i], ))




