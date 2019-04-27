import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
#print(os.listdir('./data/'))
noExamples = len(os.listdir('./data'))
data = np.zeros((noExamples,512,512))
labels = np.zeros((noExamples,1))
for itemId, file in zip(range(0,noExamples),os.listdir('./data')):
    with h5py.File('./data/'+file, 'r') as f:
        data[itemId,:,:] = f['cjdata']['image']
        labels[itemId] = f['cjdata']['label']

#print(data[0,200:210,200:210])
#plt.imshow(data[0,:,:])
#plt.show()
#print(data.shape)
#print(labels.shape)
#print(labels)
