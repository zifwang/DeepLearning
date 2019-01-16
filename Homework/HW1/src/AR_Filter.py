# Assignment: Homework1-Problem2-Filter
# Author: Zifan Wang
# Object: Simulation of first order AR Filter

import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt


# Fisrt Order AR Filter: y[n] = (1-a) - a*y[n-1]
# Experiment a with four vals.
Alpha = [0.9,0.5,0.1,-0.5]

# Plot frequency response for the AR filter
W = []
H = []
for alpha in Alpha:
    b = 1-alpha
    a = np.array([1,-alpha])
    w,h = signal.freqz(b,a,fs=1)
    W.append(w)
    H.append(h)

fig = plt.figure()
plt.title('Frequency Response')
plt.plot(W[0],20*np.log10(abs(H[0])),'r',label = 'alpha = 0.9') # alpha = 0.9
plt.plot(W[1],20*np.log10(abs(H[1])),'g',label = 'alpha = 0.5') # alpha = 0.5
plt.plot(W[2],20*np.log10(abs(H[2])),'b',label = 'alpha = 0.1') # alpha = 0.1
plt.plot(W[3],20*np.log10(abs(H[3])),'y',label = 'alpha = -0.5') # alpha = -0.5
legend = plt.legend()
plt.ylabel('Amplitude [dB]')
plt.xlabel('v (cycle/sample)')
plt.show()

# Plot Impulse Response for the AR filter
# num = [0.2]
# den = [1,-0.8]
# tf = signal.TransferFunction(num,den)
# T,yout = signal.impulse(tf, N=20)

# fig = plt.figure()
# markerline, stemlines, baseline = plt.stem(T, yout, '-')
# plt.show()

# Design an L = 4 Butterworth filter with bandwidth of ν0 = 0.25.
[b,a] = signal.butter(4,0.25,btype='low')
print('Numerator: ',b)
print('Denominator: ',a)
# Frequency response
w,h = signal.freqz(b,a,fs=1)
fig = plt.figure()
plt.title('Frequency Response')
plt.plot(w,20*np.log10(abs(h)),'r')
plt.ylabel('Amplitude [dB]')
plt.xlabel('v (cycle/sample)')
plt.show()

# Play arround with this filter
# Create a numpy array of length 300 comprising iid realizations of a standard normal distribution.
inputSignal = np.random.normal(50,1,300)
outputSignal = signal.lfilter(b,a,inputSignal)
fig = plt.figure()
plt.plot(inputSignal,'r', label='Input Signal')
plt.plot(outputSignal,'b', label='Output Signal')
plt.legend()
plt.show()