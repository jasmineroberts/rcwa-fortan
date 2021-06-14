# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:03:01 2021

@author: jasmineroberts
"""
from numpy import *
from time import time
from scipy.io import loadmat
import scipy
from copy import deepcopy
import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def Ag(lambdaX):
    A = array([[ 0.2   ,  1.072 ,  1.24  ],
       [ 0.2033,  1.098 ,  1.26  ],
       [ 0.2066,  1.125 ,  1.27  ],
       [ 0.2138,  1.173 ,  1.29  ],
       [ 0.2214,  1.208 ,  1.3   ],
       [ 0.2296,  1.238 ,  1.31  ],
       [ 0.2384,  1.265 ,  1.33  ],
       [ 0.248 ,  1.298 ,  1.35  ],
       [ 0.253 ,  1.32  ,  1.35  ],
       [ 0.2583,  1.343 ,  1.35  ],
       [ 0.2638,  1.372 ,  1.35  ],
       [ 0.2695,  1.404 ,  1.33  ],
       [ 0.2755,  1.441 ,  1.31  ],
       [ 0.2818,  1.476 ,  1.26  ],
       [ 0.2883,  1.502 ,  1.19  ],
       [ 0.2952,  1.519 ,  1.08  ],
       [ 0.2988,  1.522 ,  0.992 ],
       [ 0.3024,  1.496 ,  0.882 ],
       [ 0.3061,  1.432 ,  0.766 ],
       [ 0.31  ,  1.323 ,  0.647 ],
       [ 0.3115,  1.246 ,  0.586 ],
       [ 0.3139,  1.149 ,  0.54  ],
       [ 0.3155,  1.044 ,  0.514 ],
       [ 0.3179,  0.932 ,  0.504 ],
       [ 0.3195,  0.815 ,  0.526 ],
       [ 0.322 ,  0.708 ,  0.565 ],
       [ 0.3237,  0.616 ,  0.609 ],
       [ 0.3263,  0.526 ,  0.663 ],
       [ 0.3306,  0.371 ,  0.813 ],
       [ 0.3324,  0.321 ,  0.902 ],
       [ 0.3351,  0.294 ,  0.986 ],
       [ 0.3397,  0.259 ,  1.12  ],
       [ 0.3444,  0.238 ,  1.24  ],
       [ 0.3542,  0.209 ,  1.44  ],
       [ 0.3647,  0.186 ,  1.61  ],
       [ 0.3757,  0.2   ,  1.67  ],
       [ 0.3875,  0.192 ,  1.81  ],
       [ 0.4   ,  0.173 ,  1.95  ],
       [ 0.4133,  0.173 ,  2.11  ],
       [ 0.4275,  0.16  ,  2.26  ],
       [ 0.4428,  0.157 ,  2.4   ],
       [ 0.4592,  0.144 ,  2.56  ],
       [ 0.4769,  0.132 ,  2.72  ],
       [ 0.4959,  0.13  ,  2.88  ],
       [ 0.5166,  0.13  ,  3.07  ],
       [ 0.5391,  0.129 ,  3.25  ],
       [ 0.5636,  0.12  ,  3.45  ],
       [ 0.5904,  0.121 ,  3.66  ],
       [ 0.6199,  0.131 ,  3.88  ],
       [ 0.6526,  0.14  ,  4.15  ],
       [ 0.6888,  0.14  ,  4.44  ],
       [ 0.7293,  0.148 ,  4.74  ],
       [ 0.7749,  0.143 ,  5.09  ],
       [ 0.8266,  0.145 ,  5.5   ],
       [ 0.8856,  0.163 ,  5.95  ],
       [ 0.9537,  0.198 ,  6.43  ],
       [ 1.033 ,  0.226 ,  6.99  ],
       [ 1.127 ,  0.251 ,  7.67  ],
       [ 1.265 ,  0.375 ,  7.78  ],
       [ 1.291 ,  0.383 ,  7.92  ],
       [ 1.319 ,  0.392 ,  8.06  ],
       [ 1.348 ,  0.401 ,  8.21  ],
       [ 1.378 ,  0.411 ,  8.37  ],
       [ 1.409 ,  0.421 ,  8.37  ],
       [ 1.442 ,  0.431 ,  8.7   ],
       [ 1.476 ,  0.442 ,  8.88  ],
       [ 1.512 ,  0.455 ,  9.08  ],
       [ 1.55  ,  0.469 ,  9.32  ],
       [ 1.59  ,  0.485 ,  9.57  ],
       [ 1.631 ,  0.501 ,  9.84  ],
       [ 1.675 ,  0.519 , 10.1   ],
       [ 1.722 ,  0.537 , 10.4   ],
       [ 1.771 ,  0.557 , 10.7   ],
       [ 1.823 ,  0.578 , 11.1   ],
       [ 1.879 ,  0.6   , 11.4   ],
       [ 1.937 ,  0.624 , 11.8   ],
       [ 2.    ,  0.65  , 12.2   ],
       [ 2.066 ,  0.668 , 12.6   ],
       [ 2.138 ,  0.729 , 13.    ],
       [ 2.214 ,  0.774 , 13.5   ],
       [ 2.296 ,  0.823 , 14.    ],
       [ 2.384 ,  0.878 , 14.5   ],
       [ 2.48  ,  0.939 , 15.1   ],
       [ 2.583 ,  1.007 , 15.7   ],
       [ 2.695 ,  1.083 , 16.4   ],
       [ 2.818 ,  1.168 , 17.1   ],
       [ 2.952 ,  1.265 , 17.9   ],
       [ 3.1   ,  1.387 , 18.8   ],
       [ 3.263 ,  1.536 , 19.8   ],
       [ 3.444 ,  1.71  , 20.9   ],
       [ 3.647 ,  1.915 , 22.1   ],
       [ 3.875 ,  2.16  , 23.5   ],
       [ 4.133 ,  2.446 , 25.1   ],
       [ 4.428 ,  2.786 , 26.9   ],
       [ 4.769 ,  3.202 , 29.    ],
       [ 5.166 ,  3.732 , 31.3   ],
       [ 5.636 ,  4.425 , 34.    ],
       [ 6.199 ,  5.355 , 37.    ],
       [ 6.526 ,  5.96  , 38.6   ],
       [ 6.888 ,  6.67  , 40.4   ],
       [ 7.293 ,  7.461 , 42.5   ],
       [ 7.749 ,  8.376 , 44.8   ],
       [ 8.266 ,  9.441 , 47.1   ],
       [ 8.856 , 10.69  , 49.4   ],
       [ 9.537 , 12.21  , 52.2   ],
       [ 9.919 , 13.11  , 53.7   ]])
    lam=A[:,0]
    kap=A[:,2]
    ref=A[:,1]
    if lambdaX<lam[0] or lambdaX>lam[-1]:
            y=0
    else:
        ind=where(lam>=lambdaX)[0]
        wave_R=lam[ind[0]]
        if ind[0]==1:
            n=ref[0]
            k=kap[0]
        else:
            wave_L=lam[ind[0]-1]
            n=(ref[ind[0]]-ref[ind[0]-1])/(wave_R-wave_L)*(lambdaX-wave_L)+ref[ind[0]-1];
            k=(kap[ind[0]]-kap[ind[0]-1])/(wave_R-wave_L)*(lambdaX-wave_L)+kap[ind[0]-1];
            
    return (n+ complex(0,1) *k)**2








def Matrix_Gen_TE(e_m, e_d, f1, f2, Period, e1,lambdaX, theta, Num_ord):
    j = complex(0,1)
    ordMin=-Num_ord;
    ordMax=Num_ord;
    ordDif=2*Num_ord+1;
    i = arange(ordMin,ordMax+1)    #-40 to 40
    k0 = 2*pi/lambdaX;
    kxi = k0*(sqrt(e1)*sin(theta)+i*lambdaX/Period); #kxi (1_by_i vector)
    Kx2 = diag(power(kxi/k0,2)); #diagonal matrix with kxi/k0 (#i_by_#i matrix)

    #Fourier expansion of the dielectric function in grating
    #region: positive (1 to 80)
    h = arange(1,ordMax-ordMin+1)
    epsilonp = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(e_m-e_d);
 
    #0th order
    epsilon0 = (f2-f1)*(e_m-e_d)+e_d;
 
    #Fourier expansion of the dielectric function in grating
    #region: negative (-80 to -1)
    h = arange(ordMin-ordMax,-1+1)
    epsilonn = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(e_m-e_d);
 
    #Fourier expansion of the dielectric function in grating region
    epsilonG =hstack( [epsilonn, epsilon0, epsilonp])    #(-80 to -1, 0, 1 to 80)

    #matrix of the dielectric function of grating based on Fourier
    #components
    inE = arange(ordMax-ordMin+2,(ordMax-ordMin+2+2*Num_ord)+1 )-1;
    E = zeros([ordMax-ordMin+1,len(inE)],dtype = complex)
    for i in range(1,ordMax-ordMin+2):
        E[i-1,:] = epsilonG[ordMax-ordMin+2-i-1:ordMax-ordMin+2-i+2*Num_ord ]
    
    #Note E(1,:)=[0, ..., 80]
    #     E(2,:)=[-1, ..., 79]
    #     E(81,:)=[-80, ..., 0]

    I = eye(ordDif); #Unit matrix
    A = Kx2-E;
    Q,W =linalg.eig(A);   #Q:eigenvalue, W:corresponding vector
    Q = diag(Q)
    Q = sqrt(Q);
    V = W@Q;
    
    
    
    
    
    return Q,V,W

def RCWA_Multi_TE(N, e_m, e_d, f1, f2, Period, d, e, lambdaX, theta, Num_ord):
    j = complex(0,1)
    ordMax = Num_ord;
    ordMin = -Num_ord;
    ordDif = ordMax-ordMin+1; #total order of diffraction including 0th
    i = arange( ordMin,ordMax+1 )    #-40 to 40

    # Matrix needed for calculation
    I = eye(ordDif);
    Dirac_del=zeros([ordDif,1]);
    Dirac_del[Num_ord]=1;

    k0 = 2*pi/lambdaX;
    kxi = k0*(sqrt(e[0])*sin(theta)+i*lambdaX/Period[0]); #kxi (1_by_i vector)
    Kx = diag(kxi/k0)
    Q = zeros([N],dtype = object)
    V = zeros([N],dtype = object)
    W = zeros([N],dtype = object)
    X = zeros([N],dtype = object)
    O = zeros([N],dtype = object)
    for ind in range(N):
        Q[ind],V[ind],W[ind]=Matrix_Gen_TE(e_m[ind], e_d[ind], f1[ind], f2[ind], Period[ind], e[0], lambdaX, theta, Num_ord)
        X[ind] = diag(exp(-k0*d[ind]*linalg.eig(Q[ind])[0] ));
        O[ind] = vstack( [ hstack( [ W[ind], W[ind] ]) , hstack([ V[ind], -V[ind] ]) ] )

    kz1i = e[0]*k0*k0-kxi**2
    Kz1 =array( [cmath.sqrt(x) for x in kz1i])
    Kz3i = e[1]*k0*k0-kxi**2  
    Kz3 = array( [cmath.sqrt(x) for x in Kz3i])
    
    #From here, we solve the matrix formulation
    # TE
    Yinc = diag(Kz1/k0);   #First medium: air (or dielectric)
    Ysub = diag(Kz3/k0);   #Last medium: substrate
    
    f = zeros([N+1],dtype = object)
    g = zeros([N+1],dtype = object)
    a = zeros([N],dtype = object)
    b = zeros([N],dtype = object)
    
    f[N]=deepcopy( I )
    g[N]=j*Ysub;
    for ind in arange(N-1,-1,-1):
        mt1 = O[ind]
        mt1inv = linalg.inv(mt1)
        mat2 = vstack( [f[ind+1], g[ind+1]] )
        Mat_ab = mt1inv@ mat2
        maind = Mat_ab[0:ordDif,:]
        a[ind] = maind
        mbind = Mat_ab[ordDif:2*ordDif+1,:]
        b[ind] = mbind
        
        f[ind] = W[ind]@(I+X[ind]@a[ind]@linalg.inv(b[ind])@X[ind])
        g[ind] = V[ind]@(-I+X[ind]@a[ind]@linalg.inv(b[ind])@X[ind])
    T1 = linalg.inv(j*Yinc@f[0]+g[0])@(j*Yinc@Dirac_del+j*sqrt(e[0])*cos(theta)*Dirac_del);
    R = f[0]@T1-Dirac_del;
    T = zeros([N+1],dtype = object)

    T[0]=T1;
    for ind in range( 1 , N+1):
        T[ind] = linalg.inv(b[ind-1])@X[ind-1]@T[ind-1]
    
    Cnp = zeros([N],dtype = object)
    Cnn = zeros([N],dtype = object)
    for ind in range( N ):
        Cnp[ind]=a[ind]@linalg.inv(b[ind])@X[ind]@T[ind]
        Cnn[ind]=T[ind]
    T=T[N]
    DEr =( conj(R.T*conj(R).T*real(Kz1)).T/(k0*sqrt(e[0])*cos(theta)) )
    DEt = conj(T.T*conj(T).T*real(Kz3)).T/(k0*sqrt(e[0])*cos(theta))
   
    Ref=np.sum(DEr);
    Tran=np.sum(DEt);
    
    here = 1



    return Ref, Tran,W,V,Cnp,Cnn,Q,Kx





def Matrix_Gen_TM(e_m, e_d, f1, f2, Period, e1,lambdaX, theta, Num_ord):
    j = complex(0,1)
    ordMin=-Num_ord;
    ordMax=Num_ord;
    ordDif=2*Num_ord+1;
    i = arange(ordMin,ordMax+1)    #-40 to 40
    k0 = 2*pi/lambdaX;
    kxi = k0*(sqrt(e1)*sin(theta)+i*lambdaX/Period); #kxi (1_by_i vector)
    Kx = diag(kxi/k0); #diagonal matrix with kxi/k0 (#i_by_#i matrix)

    #Fourier expansion of the dielectric function in grating
    #region: positive (1 to 80)
    h = arange(1,ordMax-ordMin+1)
    epsilonp = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(e_m-e_d);
    epsilonp_rec = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(1/e_m-1/e_d)
    #0th order
    epsilon0 = (f2-f1)*(e_m-e_d)+e_d;
    epsilon0_rec = (f2-f1)*(1/e_m-1/e_d)+1/e_d;
    #Fourier expansion of the dielectric function in grating
    #region: negative (-80 to -1)
    h = arange(ordMin-ordMax,-1+1)
    epsilonn = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(e_m-e_d);
    epsilonn_rec = j*(exp(-j*h*2*pi*f2)-exp(-j*h*2*pi*f1))/(2*pi*h) *(1/e_m-1/e_d)
    #Fourier expansion of the dielectric function in grating region
    epsilonG =hstack( [epsilonn, epsilon0, epsilonp])    #(-80 to -1, 0, 1 to 80)
    epsilonG_rec =hstack( [epsilonn_rec, epsilon0_rec, epsilonp_rec])    #(-80 to -1, 0, 1 to 80)

    #matrix of the dielectric function of grating based on Fourier
    #components
    inE = arange(ordMax-ordMin+2,(ordMax-ordMin+2+2*Num_ord)+1 )-1;
    E = zeros([ordMax-ordMin+1,len(inE)],dtype = complex)
    E_rec = zeros([ordMax-ordMin+1,len(inE)],dtype = complex)
    for i in range(1,ordMax-ordMin+2):
        E[i-1,:] = epsilonG[ordMax-ordMin+2-i-1:ordMax-ordMin+2-i+2*Num_ord ]
        E_rec[i-1,:] = epsilonG_rec[ordMax-ordMin+2-i-1:ordMax-ordMin+2-i+2*Num_ord ]
    
    #Note E(1,:)=[0, ..., 80]
    #     E(2,:)=[-1, ..., 79]
    #     E(81,:)=[-80, ..., 0]

    I = eye(ordDif); #Unit matrix
    B = Kx@linalg.inv(E)@Kx-I
    A = linalg.inv(E_rec)@B
    Q,W =linalg.eig(A);   #Q:eigenvalue, W:corresponding vector
    Q = diag(Q)
    Q = sqrt(Q);
    V = E_rec@W@Q;
    
    
    
    
    
    return Q,V,W,E





def RCWA_Multi_TM(N, e_m, e_d, f1, f2, Period, d, e, lambdaX, theta, Num_ord):
    j = complex(0,1)
    ordMax = Num_ord;
    ordMin = -Num_ord;
    ordDif = ordMax-ordMin+1; #total order of diffraction including 0th
    i = arange( ordMin,ordMax+1 )    #-40 to 40

    # Matrix needed for calculation
    I = eye(ordDif);
    Dirac_del=zeros([ordDif,1]);
    Dirac_del[Num_ord]=1;

    k0 = 2*pi/lambdaX;
    kxi = k0*(sqrt(e[0])*sin(theta)+i*lambdaX/Period[0]); #kxi (1_by_i vector)
    Kx = diag(kxi/k0)
    Q = zeros([N],dtype = object)
    V = zeros([N],dtype = object)
    W = zeros([N],dtype = object)
    E = zeros([N],dtype = object)
    X = zeros([N],dtype = object)
    O = zeros([N],dtype = object)
    for ind in range(N):
        Q[ind],V[ind],W[ind],E[ind]=Matrix_Gen_TM(e_m[ind], e_d[ind], f1[ind], f2[ind], Period[ind], e[0], lambdaX, theta, Num_ord)
        X[ind] = diag(exp(-k0*d[ind]*linalg.eig(Q[ind])[0] ));
        O[ind] = vstack( [ hstack( [ W[ind], W[ind] ]) , hstack([ V[ind], -V[ind] ]) ] )

    kz1i = e[0]*k0*k0-kxi**2
    Kz1 =array( [cmath.sqrt(x) for x in kz1i])
    Kz3i = e[1]*k0*k0-kxi**2  
    Kz3 = array( [cmath.sqrt(x) for x in Kz3i])
    
    #From here, we solve the matrix formulation
    # TE
    Yinc = diag(Kz1/k0);   #First medium: air (or dielectric)
    Ysub = diag(Kz3/k0);   #Last medium: substrate
    
    f = zeros([N+1],dtype = object)
    g = zeros([N+1],dtype = object)
    a = zeros([N],dtype = object)
    b = zeros([N],dtype = object)
    
    f[N]=deepcopy( I )
    g[N]=j*Ysub;
    for ind in arange(N-1,-1,-1):
        mt1 = O[ind]
        mt1inv = linalg.inv(mt1)
        mat2 = vstack( [f[ind+1], g[ind+1]] )
        Mat_ab = mt1inv@ mat2
        maind = Mat_ab[0:ordDif,:]
        a[ind] = maind
        mbind = Mat_ab[ordDif:2*ordDif+1,:]
        b[ind] = mbind
        
        f[ind] = W[ind]@(I+X[ind]@a[ind]@linalg.inv(b[ind])@X[ind])
        g[ind] = V[ind]@(-I+X[ind]@a[ind]@linalg.inv(b[ind])@X[ind])
    T1 = linalg.inv(j*Yinc@f[0]+g[0])@(j*Yinc@Dirac_del+j*sqrt(e[0])*cos(theta)*Dirac_del);
    R = f[0]@T1-Dirac_del;
    T = zeros([N+1],dtype = object)

    T[0]=T1;
    for ind in range( 1 , N+1):
        T[ind] = linalg.inv(b[ind-1])@X[ind-1]@T[ind-1]
    
    Cnp = zeros([N],dtype = object)
    Cnn = zeros([N],dtype = object)
    for ind in range( N ):
        Cnp[ind]=a[ind]@linalg.inv(b[ind])@X[ind]@T[ind]
        Cnn[ind]=T[ind]
    T=T[N]
    DEr =( conj(R.T*conj(R).T*real(Kz1)).T/(k0*sqrt(e[0])*cos(theta)) )
    DEt = conj(T.T*conj(T).T*real(Kz3)).T/(k0*sqrt(e[0])*cos(theta))
   
    Ref=np.sum(DEr);
    Tran=np.sum(DEt);
    
    here = 1



    return Ref,Tran,W,V,Cnp,Cnn,E,Q,Kx