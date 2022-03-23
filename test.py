import nibabel as nib
import scipy
import numpy as np
import cv2
from scipy import ndimage
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import matplotlib.pyplot as plt


def noisy(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

def noisy2(image):
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

def noisy3(image):
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy

def noisy4(image):
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

#img = nib.load('/home/gkarozis/adni/ADNI/002_S_0559/HarP_135_final_release_2015/2009-06-30_17_01_28.0/S69303/ADNI_002_S_0559_MR_HarP_135_final_release_2015_Br_20150226103031117_S69303_I474785.nii')
#img = nib.load('/home/gkarozis/adni/AD/ADNI_130_S_0956_MR_MPR____N3_Br_20081209120533395_S58319_I129654.nii')
img = nib.load('/home/gkarozis//adni/ADNI/031_S_0830/MPR____N3/2008-09-25_11_10_07.0/S56750/ADNI_031_S_0830_MR_MPR____N3_Br_20090128130741576_S56750_I134800.nii')

data = img.get_fdata()
print('DATA SHAPE1', data.shape)
#data = data.swapaxes(0,2)
#data = data.swapaxes(0,1)
#IN CASE OF NOT PREPROCESSED DATA
#height, width, depth, channels = data.shape
data = data[40:150,50:210,35:120]
print('DATA SHAPE2', data.shape)

height, width, depth = data.shape

print('The image has the following dimensions: \nheight:='+str(height)+'\nwidth:='+str(width)
       +'\ndepth ='+str(depth))
print('Plotting layer 80')
plt.imshow(data[:, :, 0],cmap='gray')
plt.axis('off')
plt.show()
'''
shift_image = scipy.ndimage.shift(data, np.array([2, 2, 1]))

height, width, depth = shift_image.shape
'''
#gaussian_noise_image = noisy2(data)
#height, width, depth = gaussian_noise_image.shape
'''
print('The image has the following dimensions: \nheight:='+str(height)+'\nwidth:='+str(width)
       +'\ndepth ='+str(depth))
print('Plotting layer 80')
plt.imshow(shift_image[:, :, 50],cmap='gray')
plt.axis('off')
plt.show()
'''
'''
data2 = ndimage.rotate(data, 8, reshape=False)
height, width, depth = data2.shape
print('The image has the following dimensions: \nheight:='+str(height)+'\nwidth:='+str(width)
       +'\ndepth ='+str(depth))
plt.imshow(data2[:, :, 50],cmap='gray')
plt.axis('off')
plt.show()
'''