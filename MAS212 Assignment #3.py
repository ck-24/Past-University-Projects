# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 23:17:32 2018

@author: charl
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Slide 2
data = np.loadtxt('pulsar_signal.txt') 
t = data[:,0]
y = data[:,1]
plt.plot(t, y)
plt.xlabel('t / sec'); plt.ylabel('signal y(t)')

###Finding the average signal
avy = []
for n in range(189):
    pulse = []
    for i in range(10):
        pulse.append(y[n+i*189])
    avy.append(np.mean(pulse))

fig = plt.figure()
t = t[0:189]
plt.plot(t, avy)
plt.xlabel('t / sec'); plt.ylabel('signal y(t)')

        
#Slide 4
###Solving the normal equations to fit the linear model
N = len(t)
m = 19 
X = np.zeros(shape=(N, m))
for i in range(N):
    X[i,:] = np.array([t[i]**k for k in range(m)])


A = np.dot(X.transpose(), X)
b = np.dot(X.transpose(), avy)
par = np.linalg.solve(A, b)
print("Beta 0,1 = ", par[0],par[1])

n = 189
yline = np.zeros(n)
for i in range(m):
    yline += par[i]*t**i
fig = plt.figure()
plt.plot(t, avy, 'b-')
plt.plot(t, yline, 'r-', lw=3)
plt.xlabel('t / sec'); plt.ylabel('signal y(t)')
 

#Slide 5

def f(t, c, A, tau, sigma):
    return c + A*np.exp(-(t-tau)**2/(2*(sigma)**2))


par0 = [0, 1, 0, 0.1]
par, cov = curve_fit(f, t, avy, par0)
print("Parameters : ", par)
n = 189
yline2 = np.zeros(n)
for i in range(n):
    yline2[i] = f(t[i], par[0],par[1],par[2],par[3])
fig = plt.figure()
plt.plot(t, avy, '.')
plt.plot(t, yline2, 'r-', lw=3)
plt.xlabel('t / sec'); plt.ylabel('signal y(t)');

#Slide 6

def rmsd(r):
    N = len(r)
    return (np.sum(r**2)/N)**0.5

print("RMSD of the linear model is", rmsd(avy-yline))
print("RMSD of the non-linear model is", rmsd(avy-yline2))

 
fig=plt.figure()
plt.plot((avy - yline)[0:188], (avy - yline)[1:189], '.')
plt.xlabel('$r_k$'); plt.ylabel('$r_{k+1}$');

fig=plt.figure()
plt.plot((avy - yline2)[0:188], (avy - yline2)[1:189], '.')
plt.xlabel('$r_k$'); plt.ylabel('$r_{k+1}$');


 

 
 