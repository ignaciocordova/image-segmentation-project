#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:26:41 2022
Final Project for the course Image Processing and Artificial Vision, UB, 2022

Please refer to the Jupyter Notebooks for a better explanation of the code
and of the process. 

@author: Ignacio Córdova 

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import skimage
from scipy.ndimage.measurements import histogram
from sklearn.cluster import KMeans
from scipy import ndimage
import impavi 
import seaborn as sn
import pandas as pd

"""
Image Segmentation techniques ( 1/3 )

The first image I will work with is the easiest one for segmentation. This part
 of the study will provide a general idea of how segmentation can be performed
 and will serve as introduction to better understand the next, more problematic
 images. As you can see in the image below, **segmentation can not be performed 
 based on color (or gray-scale value)** so I will have to find other properties 
 that vary between regions.

"""

im = plt.imread('DIC_SN_15_L10_Sum00.tif')
im_eq = impavi.equalize(im,plot=False)

"""
High pass filter aproach
 
 The idea behind this kind of segmentation relies on the structure difference 
 between the central cut and the tissue. I will exploit the texture properties 
 of the membrane to differentiate it from the wound.
 
 One possible approach is to try to retain the parts of the image with high 
 detail. In other words, gather in formation aboout change of contrast, edges,
 shapes, etc. This can be achieved by applying a "high pass filter" to the 
 Fourier Transform of the image. This means dropping the low frequencies in the
 Fourier space and keeping just the frequencies in charge of these properties
 (change of contrast, edges, shapes, etc).
"""



#Fast Fourier Transform
im_tf = np.fft.fftshift(np.fft.fft2(im_eq))
#build Laplacian Filter 
u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
lf = u**2 + v**2
circ = lf>0.005

im1 = np.abs(np.fft.ifft2(im_tf*circ))

median = ndimage.median_filter(im1, size=[11,41])


# second MEDIAN THRESHOLD
# media = scipy.signal.medfilt(im, [mask size])
median2 = scipy.signal.medfilt2d(median, kernel_size=[11,41])


## K-means clustering

"""
The next function I built (can be found in the "impavi" module) performs
 K-means to the image using k=7 in this case. This allows to have a better,
 more refined segmentation than just a binarization. I built he function 
 "apply_kmeans" so that it automatically returns the largest cluster out 
 of the k clusters. 
 """
im_seg = impavi.apply_kmeans(median2,7,plot=True)

plt.figure()
plt.axis('off')
plt.title('Final Result using image 1/3')
plt.imshow(im_seg,cmap='gray')


"""
The next function computes SSIM between the target and also of all possible 
combinations between my result and the state of the art algorithms' results. 
This provides a correlation matrix that can be represented with highly visual
 information about the performance of each one of the results.
"""

ssim = impavi.ssim_matrix('DIC_SN_15_L10_Sum00',im_seg)


sn.set(font_scale=2)

df_cm = pd.DataFrame(ssim, index = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']],
                  columns = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,fmt='.3g',)



#%%

"""
Image Segmentation techniques ( 2/3 )

The second image I will work with has a high structural complexity of the edges.
 This patterns will cause problems to the state of the art algorithms. This
 part of the study will provide a new way of obtaining features that can easily 
 differentiate the two regions (cut and tissue). As you can see in the image 
 below, **segmentation can not be performed based on color** so I will have to 
 find other properties that vary between regions. 
"""

im = plt.imread('DIC_SN_15_L7_Sum00.tif')
im_eq = impavi.equalize(im,plot=False)

"""
Edge enhancement approach 
KIRSCH COMPASS KERNEL

I decided to apply this a technique which will exploit the edges of the tissue 
(and the absence of edges of the cut). This algorithm published in by *Kirsch,
 R. (1971)* in "*Computer determination of the constituent structure of
 biological images*" will give a very useful result to later apply the median
 filter.
"""

im_kirsch = impavi.kirsch_compass_kernel(im_eq)

media = scipy.signal.medfilt2d(im_kirsch, kernel_size=11)

media2 = scipy.signal.medfilt2d(media, kernel_size=[31,31])

final = media2>skimage.filters.threshold_otsu(media2)*0.75

plt.figure(figsize=(14,14))
plt.subplot(121)
plt.axis('off')
plt.imshow(im,cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.imshow(final,cmap='gray')


ssim = impavi.ssim_matrix('DIC_SN_15_L7_Sum00',final)


sn.set(font_scale=2)

df_cm = pd.DataFrame(ssim, index = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']],
                  columns = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,fmt='.3g',)


#%%

"""
Image Segmentation techniques ( 3/3 )

This last problem is a very interesting one involving very poor lighting
 conditions. I will apply very specific filters and transformations that will 
 manage to perform the segmentation improving the SSIM. As you can see in the 
 image below, segmentation can not be performed based on color (or gray-scale
 value) so I will have to find other properties that vary between regions.
"""

im = plt.imread('DIC_SN_15_L38_Sum16.tif')
im_eq = impavi.equalize(im,plot=False)

"""
High pass filter aproach

As in the first part, one possible approach is to try to retain the parts 
of the image with high detail. In other words, gather information aboout change
 of contrast, edges, shapes, etc. This can be achieved by applying a "high pass
 filter" to the Fourier Transform of the image. Varying the radius of the sharp
 cut-off filter we can obtain different results. The best one is obtained by
 using a radius of 0.005 and the image is shown below.
"""

#Fast Fourier Transform
im_tf = np.fft.fftshift(np.fft.fft2(im_eq))
#build Laplacian Filter 
u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
lf = u**2 + v**2
circ = lf>0.005

im1 = np.abs(np.fft.ifft2(im_tf*circ))

"""
I proceed to average the image with a very specific filter. That is, a 
rectangular filter with a higher dimension along the horizontal axis. This will
 minimize the loss of information at the edges of the cut and will get rid of
 the noise (white spots) on the cut region. 
"""
media = scipy.signal.medfilt2d(im1, kernel_size=[3,201])

#To obtain the final result, I proceed to binarize the image and 
#apply two more median filters.

final = (media>0.04).astype(float)
media = scipy.signal.medfilt2d(final, kernel_size=[11,201])

media2 = scipy.signal.medfilt2d(media, kernel_size=[11,201])

"""
This time, a low-pass filter allows us to eliminate noise and smooth the 
result. Note that this is the complementary filter to the one previously used
 (the effect obtained is opposite). 
"""
#Fast Fourier Transform
im_tf = np.fft.fftshift(np.fft.fft2(media2))
#build Laplacian Filter 
u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
lf = u**2 + v**2
circ = lf<0.0009

im1 = np.abs(np.fft.ifft2(im_tf*circ))

final = im1>0.2
plt.figure(figsize=(12,12))
plt.axis('off')
plt.imshow(final,cmap='gray')

ssim = impavi.ssim_matrix('DIC_SN_15_L38_Sum16',final)

sn.set(font_scale=2)

df_cm = pd.DataFrame(ssim, index = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']],
                  columns = [i for i in ['Manual','Topman','TScratch','MultiCellSeg','My result']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,fmt='.3g',)