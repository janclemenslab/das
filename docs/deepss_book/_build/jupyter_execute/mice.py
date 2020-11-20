#!/usr/bin/env python
# coding: utf-8

# # Ultrasonic vocalizations from mice

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa.feature
import colorcet as cc
import umap
from scipy.cluster.vq import kmeans2
import deepss_unsupervised.spec_utils
from io import BytesIO
import urllib.request

plt.style.use('ncb.mplstyle')


# In[2]:


# import dss.npy_dir
# data = dss.npy_dir.load('/Volumes/shinkansen/Dropbox/dss.paper/dat/mice_06.npy')
# fs = data.attrs['samplerate_x_Hz']
# x = np.concatenate((data['train']['x'][:, 0], data['test']['x'][:, 0], data['val']['x'][:, 0]))
# y = np.concatenate((data['train']['y'][:, 1], data['test']['y'][:, 1], data['val']['y'][:, 1]))
# diff = np.diff((y>0).astype(np.double))

# onsets = np.where(diff==1)[0]
# offsets = np.where(diff==-1)[0]
# np.savez_compressed('data/mice.npz', recording=x, samplerate=fs, 
#                     syllable_onsets=onsets.astype(np.intp),
#                     syllable_offsets=offsets.astype(np.intp))  # data by K. Hammerschmidt


# In[3]:


# data provided by Kurt Hammerschmidt and published in A Ivanenko, P Watkins, MAJ van Gerven, K Hammerschmidt, B Englitz (2020) Classifying sex and strain from mouse ultrasonic vocalizations using deep learning. PLoS Comput Biol 16(6): e1007918.
url = 'https://github.com/janclemenslab/deepss_unsupervised/releases/download/v0.4/mice.npz'
with urllib.request.urlopen(url) as f:
    ff = BytesIO(f.read())

d = np.load(ff)
samplerate = d['samplerate']
recording = d['recording']
syllable_onsets = d['syllable_onsets']
syllable_offsets = d['syllable_offsets']


# In[4]:


n_fft = 2048
hop_length = int(samplerate / 1000)

specgram = librosa.feature.melspectrogram(recording, sr=samplerate, n_fft=n_fft, hop_length=hop_length, power=1, fmin=40_000)
freqs = librosa.core.mel_frequencies(n_mels=specgram.shape[0], fmin=40_000, fmax=int(samplerate/2), htk=False)
sm = np.median(specgram, axis=1)
print(sm.shape)


# In[5]:


plt.figure(figsize=(20, 4))
plt.imshow(np.log2(specgram[:, int(syllable_onsets[1]/hop_length)-10:int(syllable_offsets[4]/hop_length)+10]), cmap='cet_CET_L17')
for onset, offset in zip(syllable_onsets[1:5], syllable_offsets[1:5]):
    plt.axvline((onset-syllable_onsets[1])/hop_length + 10, c='k')
    plt.axvline((offset-syllable_onsets[1])/hop_length + 10, c='k')


# In[6]:


specs = [] 

plt.figure(figsize=(10, 10))
for cnt, (onset, offset) in enumerate(zip(syllable_onsets, syllable_offsets)):
    spec = np.log2(specgram[:, int(onset/hop_length):int(offset/hop_length)]/ sm[:,np.newaxis])
    spec = spec[15:-10, :]
    spec[spec<0] = 0
    specs.append(spec)
    try:
        plt.subplot(4,4,cnt+1)
        plt.imshow(spec, cmap='jet')
        plt.xlim(0, 50)
        plt.colorbar()
    except:
        pass


# In[7]:


spec_rs = [deepss_unsupervised.spec_utils.log_resize_spec(spec, scaling_factor=8) for spec in specs]

max_len = np.max([spec.shape[1] for spec in spec_rs])
print(max_len)
spec_rs = [deepss_unsupervised.spec_utils.pad_spec(spec, pad_length=max_len) for spec in spec_rs]
plt.figure(figsize=(10, 10))
for cnt, spc in enumerate(spec_rs[:16]):
    plt.subplot(4,4,cnt+1)
    plt.imshow(spc, cmap='turbo')
    plt.xlim(0, 50)


# In[8]:


offset = 20
spec_rs_c = [deepss_unsupervised.spec_utils.center_spec(spec, halfwidth=offset) for spec in spec_rs]
spec_rs_c = np.array(spec_rs_c)


# In[9]:


spec_flat = [spec.ravel() for spec in spec_rs_c]
spec_flat = np.array(spec_flat)


# In[10]:


out = umap.UMAP(min_dist=0.5).fit_transform(spec_flat)


# In[11]:


plt.figure(figsize=(3, 3))
plt.scatter(out[:,0], out[:,1], c='k', alpha=0.25, s=8)
plt.axis('tight')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()


# In[12]:


# partition umap using kmeans
k_centroid, k_labels = kmeans2(out, k=40)


# In[13]:


plt.figure(figsize=(3, 3))
plt.scatter(out[:,0], out[:,1], c=k_labels, cmap='cet_glasbey_dark', alpha=0.5, s=8, edgecolor='none')
plt.axis('tight')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()


# In[14]:


fig = plt.figure(figsize=(12, 12))
ax = plt.axes()
for cnt, (centroid, label) in enumerate(zip(k_centroid, np.unique(k_labels))):
    idx = np.where(k_labels==label)[0]
    X = np.mean(spec_rs_c[idx], axis=0)
    fig.figimage(X=X, xo=centroid[0]*40, yo=centroid[1]*40, cmap='cet_CET_L17')


# In[ ]:




