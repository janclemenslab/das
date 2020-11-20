#!/usr/bin/env python
# coding: utf-8

# # Fly song

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import deepss_unsupervised.event_utils
from io import BytesIO
import urllib.request

plt.style.use('ncb.mplstyle')


# In[2]:


# data from D Stern (2014). Reported Drosophila courtship song rhythms are artifacts of data analysis. BMC Biology. https://www.janelia.org/lab/stern-lab/tools-reagents-data
url = 'https://github.com/janclemenslab/deepss_unsupervised/releases/download/v0.4/flies.npz'
with urllib.request.urlopen(url) as f:
    ff = BytesIO(f.read())

d = np.load(ff)
recording = d['recording']
pulsetimes = d['pulsetimes']
sinetimes = d['sinetimes']
samplerate = d['samplerate']


# In[3]:


win_hw = 100
pulseshapes = deepss_unsupervised.event_utils.get_shapes(pulsetimes + win_hw, recording, win_hw)
pulseshapes = np.apply_along_axis(deepss_unsupervised.event_utils.normalize, axis=-1, arr=pulseshapes.T).T
pulseshapes = pulseshapes[25:-25]

sineshapes = deepss_unsupervised.event_utils.get_shapes(sinetimes + 2*win_hw, recording, win_hw)
sineshapes = np.apply_along_axis(deepss_unsupervised.event_utils.normalize, axis=-1, arr=sineshapes.T).T
sineshapes = sineshapes[25:-25]

good_shapes = np.sum((sineshapes==0).astype(np.float), axis=0)<10
sineshapes = sineshapes[:, good_shapes]

allshapes = np.concatenate((pulseshapes, sineshapes), axis=1)
print('Pulses:', pulseshapes.shape, 'sines:', sineshapes.shape, 'all:', allshapes.shape)
good_shapes = np.sum((sineshapes==0).astype(np.float), axis=0)<10
sineshapes = sineshapes[:, good_shapes]

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.plot(pulseshapes, linewidth=0.5, alpha=0.2)
plt.title('Pulses')

plt.subplot(122)
plt.plot(sineshapes, linewidth=0.5, alpha=0.2)
plt.title('Sines')
plt.show()


# In[4]:


out = umap.UMAP(min_dist=0.5).fit_transform(allshapes.T)
hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=20).fit_predict(out)


# In[5]:


cols = 'rgb'

plt.figure(figsize=(10, 5))
for label in np.unique(hdbscan_labels):
    if label>=0:
        this = hdbscan_labels==label
        
        plt.subplot(121)
        plt.scatter(out[this,0], out[this,1], c=cols[label], s=8, alpha=0.5, edgecolor='none')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        plt.subplot(222)
        plt.plot(allshapes[:, this], c=cols[label], alpha=0.2, linewidth=0.1)
        plt.ylim(-0.5, 0.5)
        plt.title('Individual shapes')
        
        plt.subplot(224)
        plt.plot(np.mean(allshapes[:, this], axis=1), c=cols[label])
        plt.ylim(-0.25, 0.3)
        plt.title('Centroid shapes')
        
plt.axhline(0, c='k')


# In[ ]:




