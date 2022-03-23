import nibabel as nib
import nibabel.processing
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import pandas as pd
import glob, os
from sklearn import preprocessing
import cv2
from nilearn.image import resample_img

from skimage.transform import resize
from pathlib import Path

df = pd.read_csv(r'~/adni/code/22_07_2021_8_23_2021.csv')
print(df)
no_shapes = []
max = 0

REG_DB = '/home/gkarozis/adni/'
REG_DB_SUBFOLDERS = ['AD/','MCI/','CN/']


def resample_and_save(filename, path):
    ''' Process the image name and copy the image to its
        corresponding destination folder.

        Parameters:
            filename -- Name of the image file (.nii)
            path -- The path were the image is located
    '''
    # separate the name of the file by '_'
    splitted_name = filename.strip().split('_')
    # sometimes residual MacOS files appear; ignore them
    if splitted_name[0] == '.': return

    # save the image ID
    image_ID = splitted_name[-1][0:-4]
    # sometimes empty files appear, just ignore them (macOS issue)
    if image_ID == '': return
    # transform the ID into a int64 numpy variable for indexing

    #### IMPORTANT #############
    # the following three lines are used to extract the label of the image
    # ADNI data provides a description .csv file that can be indexed using the
    # image ID. If you are not working with ADNI data, then you must be able to
    # obtain the image label (AD/MCI/NC) in some other way
    # with the ID, index the information we need
    row_index = df.index[df['Image Data ID'] == image_ID].tolist()[0]
    # obtain the corresponding row in the dataframe
    row = df.iloc[row_index]
    # get the label
    label = row['Group']
    # prepare the origin path
    complete_file_path = os.path.join(path, filename)
    # load sitk image
    nib_moving = nib.load(complete_file_path)
    data = nib_moving.get_fdata()
    if data.shape==(256, 256, 170):
       print(data.shape)
       nib_moving2 = nib.Nifti1Image(data[40:150,50:210,32:120], nib_moving.affine)
       # I chose to remove the skull and have only the information of the brain
       complete_new_path = os.path.join(REG_DB,
                                     label,
                                     filename)
       nib.save(nib_moving2, complete_new_path)


file_list = []
path_list = []
z = 'OK'
k=0
j=0
max = 0
for r, d, f in os.walk('../ADNI'):
    if f != []:
        for i in f:
           k = k + 1
           img = nib.load('/home/gkarozis/adni/'+r[3:]+'/'+i)
           data = img.get_fdata()
           try:
               resample_and_save(i, r)
           except:
               j=j+1
           file_list.append(i)
           path_list.append(r)

