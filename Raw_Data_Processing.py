import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.ndimage as ndimg

#text = "dataset/human_falling/11_3_2022_3.txt" # test2.txt is of me walking towards the UWB and then falling and then getting back up at the very end.
i = 40
#_type = "human_walking"
_type = "human_falling"
#_type = "human_limping"
text = "dataset/%s/11_6_2022_%i.txt"%(_type, i)
rawData = np.loadtxt(text)
l = rawData.shape[0]

I_Data = rawData[0:l//2,:]
Q_Data = rawData[(l//2)+1:,:]
IQ_Data = I_Data + (1j*Q_Data)
IQ_Data = np.abs(IQ_Data)
IQ_Data = IQ_Data.transpose()
print(IQ_Data.shape)

#plt.figure()
#plt.imshow(IQ_Data)

# STATIC SET FOR PRF_DIV=4
# I was messnig with the RADAR stats to match the PRF listed in the gesture paper:
#IQ_Data = IQ_Data[:,0:315]

# clutter removal? (it works kinda)
IQ_Data = (IQ_Data - np.min(IQ_Data))/np.ptp(IQ_Data)
clutter_bg = None
alpha = 0.1
clutter_bg = np.zeros_like(IQ_Data)
clutter_bg[0,:] = IQ_Data[0,:]

for k in range(1,IQ_Data.shape[0]):
    clutter_bg[k,:] = (alpha*clutter_bg[k-1,:])+((1-alpha)*IQ_Data[k,:])

IQ_Data = IQ_Data - clutter_bg # looks similar to figure 13 

# Histogram to remove sparse, super low values and recalib values
test = np.histogram(IQ_Data.ravel(), bins=50)
new_low_thres = test[1][np.argmax(test[0])]
IQ_Data = np.clip(IQ_Data, new_low_thres, IQ_Data.max())
filt = np.ones((5,5),dtype=np.float64)

# something i took from MVDR code:
IQ_Data = ndimg.correlate(IQ_Data, filt, mode='constant')

plt.figure()
plt.imshow(IQ_Data)
plt.xlabel("Fast Time")
plt.ylabel("Slow Time")
plt.show()

'''
bin_length = 8 * 1.5e8/23.328e9
max_range = 192*bin_length
IQ_Data[0:11,:] = 0; # how to determine if filtered IQ data should be 0 to 11

test = np.fft.ifft(IQ_Data, axis=1)
print(test.shape)

test2 = np.fft.fft(IQ_Data, axis=1)

plt.figure()
plt.imshow(test2.real)
plt.show()
'''