# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:28:43 2021

@author: jasmineroberts
"""
from func2File import *

tic = time()
c0 = 299792458;                             # speed of light in space, [m/s]
theta =0;                                           # angle of incidence, [rad]
wn = linspace(3500,25000,300) ; # wavenumber, [1/cm]
lambdaX =1e4/wn;                           # wavelength, [um]
w = 1e6*2*pi*c0/lambdaX;            # angular frequency, [rad/s]
d =np.array( [0.4])                                           # thickness of each layer from front to back, [um]

N = len(d);         
Period =zeros([N])
e_m = zeros([N],dtype = complex)
e_d = zeros([N],dtype = complex)

Period[0:N] =0.5;                             # Period of gratings for each layer, [um]
width = 0.45;                                    # width of metal strips, [um]
psi = width/Period[0]                     # filling ratio of grating layer
f1 = np.array([0])                                              # normalized position for left-end of metal strip
f2 =np.array( [psi])                                           # normalized position for right-end of metal strip
Num_ord = 50;  
e = np.array([0,0])
i = complex(0,1)


Ref = zeros([len(lambdaX)],dtype = complex)
Tran = zeros([len(lambdaX)],dtype = complex)
for ind in range(len(lambdaX)):
    # Incidence medium
    e[0] = 1;                                                 # Usually is air or vacuum 
    # Layered structure
    e_m[0:N] = Au(lambdaX[ind]);   # Ridge material (metal)  
    e_d[0:N] = 1+i*1e-12;                       # Groove material (air)
    #Substrate
    e[1]=1;                                                    #  air or opaque substrate
    #==========================================
    
    Ref[ind], Tran[ind] = RCWA_Multi_TE(N, e_m, e_d, f1, f2, Period, d, e, lambdaX[ind], theta, Num_ord); 

plt.plot(wn,Ref)
plt.plot(wn,Tran)
plt.plot(wn,1-Ref-Tran)
plt.legend(['Ref','Tran','alpha']);
plt.xlim([3500,25000]);
plt.xlabel('Wavenumber, nu (cm^-1)');
plt.ylabel('R,T,\alpha');
plt.show()