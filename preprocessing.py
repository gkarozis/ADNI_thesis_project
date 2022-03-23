import os
import nibabel as nib
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import time
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as score2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv3D, Dropout, PReLU
import keras.utils
import matplotlib.pyplot as plt
start_time = time.time()


#έτσι ελέγχουμε ότι τρέχουμε στη GPU
def noisy2(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

import tensorflow as tf



REG_DB = '/home/gkarozis/adni/MCI'
REG_DB2 = '/home/gkarozis/adni/CN'
REG_DB3 = '/home/gkarozis/adni/AD'
path = '/home/gkarozis/adni/MCI'
#example_filename = os.path.join(path, 'ADNI_136_S_1227_MR_MPRAGE_3dtferepeat_br_raw_20090324134324131_1_S64958_I139690.nii')
#img = nib.load(example_filename)

#a = np.array(img.dataobj)

MCI_list = []
AD_list = []
CN_list = []
a2_list = []
j=0
MCI_unique = []
AD_unique = []
CN_unique = []
max = 0
for r, d, f in os.walk('../MCI'):
      for i in f:
           splitted_name = i.strip().split('_')
           name = splitted_name[-2]
           complete_new_path = os.path.join(REG_DB,i)
           #img = nib.load(complete_new_path)
           img = nib.load(complete_new_path)
           a = img.get_fdata()
           print('MCI', a.shape)
           if name not in MCI_unique:
              j = j + 1
              if j<=40:
                  a2 = ndimage.rotate(a, 8, reshape=False)
                  a3 = ndimage.rotate(a, 352, reshape=False)
                  MCI_list.append(a2)
                  MCI_list.append(a3)
              elif j>40 and j<=90:
                  a4 = ndimage.shift(a, np.array([2, 2, 1]))
                  MCI_list.append(a4)
              elif j>90 and j<=120:
                  a5 = noisy2(a)
                  MCI_list.append(a5)
              MCI_unique.append(name)
           MCI_list.append(a)

j = 0
for r, d, f in os.walk('../AD'):
      for i in f:
           splitted_name = i.strip().split('_')
           name = splitted_name[-2]
           complete_new_path = os.path.join(REG_DB3,i)
           img = nib.load(complete_new_path)
           a = img.get_fdata()
           print('AD', a.shape)
           if name not in AD_unique:
              j = j+1
              a2 = ndimage.rotate(a, 8, reshape=False)
              a3 = ndimage.rotate(a, 352, reshape=False)
              a4 = ndimage.shift(a, np.array([2, 2, 1]))
              a5 = noisy2(a)
              AD_list.append(a2)
              AD_list.append(a3)
              if j<50:
                print(j)
                AD_list.append(a4)
                AD_list.append(a5)
              AD_unique.append(name)
           AD_list.append(a)

j=0
for r, d, f in os.walk('../CN'):
      for i in f:
           splitted_name = i.strip().split('_')
           name = splitted_name[-2]
           complete_new_path = os.path.join(REG_DB2,i)
           img = nib.load(complete_new_path)
           a = img.get_fdata()
           print('CN', a.shape)
           ###########     DATA AUGMENTATION      ####################
           if name not in AD_unique:
              j = j + 1
              if j<=40:
                  a2 = ndimage.rotate(a, 10, reshape=False)
                  a3 = ndimage.rotate(a, 350, reshape=False)
                  CN_list.append(a2)
                  CN_list.append(a3)
              if j>40 and j<=90:
                  a4 = ndimage.shift(a, np.array([2, 2, 1]))
                  CN_list.append(a4)
              if j >90 and j<=110:
                  a5 = noisy2(a)
                  CN_list.append(a5)
              CN_unique.append(name)
           CN_list.append(a)

print('MCI_UNIQUE',len(MCI_unique))
print('MCI_LIST',len(MCI_list))
print('AD_UNIQUE',len(AD_unique))
print('AD_LIST',len(AD_list))
print('CN_UNIQUE',len(CN_unique))
print('CN_LIST',len(CN_list))

print('LISTS SIZE', len(AD_list),len(MCI_list), len(CN_list))
'''
for i in AD_list:
    print('AD_LIST SHAPE', i.shape)
for i in CN_list:
    print('CN_LIST SHAPE', i.shape)
for i in MCI_list:
    print('MCI_LIST SHAPE', i.shape)
'''

images = AD_list + MCI_list + CN_list
labels = [0]*len(AD_list)+[1]*len(MCI_list)+[2]*len(CN_list)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=1)
k1=0 #AD count
k2=0 #MCI count
k3=0 #CN count
'''
data_augmented_images = []
data_augmented_labels = []
'''
print('LEN 1', len(train_images))
###########     DATA AUGMENTATION      ####################
'''
k=0
for i in train_images:
    print(train_labels[k1+k2+k2][0],train_labels[k][0], (k1+k2+k2), k)
    if train_labels[k]==0:
        k1=k1+1
    elif train_labels[k]==1:
        k2=k2+1
    elif train_labels[k]==2:
        k3=k3+1
    a2 = ndimage.rotate(i, 10, reshape=False)
    a3 = ndimage.rotate(i, 350, reshape=False)
    a4 = ndimage.shift(i, np.array([2, 2, 1]))
    a5 = noisy2(a)
    data_augmented_images.append(a2)
    data_augmented_images.append(a3)
    data_augmented_images.append(a4)
    data_augmented_images.append(a5)
    data_augmented_labels.append(train_labels[k])
    k=k+1

print(k1,k2,k3)

train_images = train_images + data_augmented_images
train_labels = train_labels + data_augmented_labels
print('LEN 2', len(train_images))
'''
class_names = ['AD','MCI','CN']
print('BINCOUNT TRAIN',np.bincount(train_labels))
print('BINCOUNT TEST',np.bincount(test_labels))
print('BINCOUNT VALIDATION', np.bincount(val_labels))
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
train_labels = to_categorical(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
test_labels = to_categorical(test_labels)
val_images = np.asarray(val_images)
val_labels = np.asarray(val_labels)
val_labels = to_categorical(val_labels)
print(np.max(train_images))
print(np.min(train_images))
'''
m = np.max(train_images)
mi = np.min(train_images)
m2 = np.max(test_images)
mi2 = np.min(test_images)
m3 = np.max(val_images)
mi3 = np.min(val_images)
'''
#print('MAX, MIN', m, mi)
#print(train_images)
'''
train_images = (train_images-mi)/(m - mi)
test_images = (test_images - mi2) / (m2 - mi2)
val_images = (val_images - mi3) / (m3 - mi3)
'''
print('MAX2, MIN2', np.max(train_images), np.min(train_images))
train_images = train_images.reshape((train_images.shape[0], 110, 160, 88, 1)) #tensorflow needs number of channels in last parameter,which due to grayscaling is 1
#print('TRAIN_LABELS', train_labels)
#print('TEST_LABELS',test_labels)
test_images = test_images.reshape((test_images.shape[0], 110, 160, 88, 1))
val_images = val_images.reshape((val_images.shape[0], 110, 160, 88, 1))
with open('/home/gkarozis/adni/code/train_images.npy', 'wb') as f:
     np.save(f, train_images)
with open('/home/gkarozis/adni/code/train_labels.npy', 'wb') as f:
    np.save(f, train_labels)
with open('/home/gkarozis/adni/code/test_images.npy', 'wb') as f:
    np.save(f, test_images)
with open('/home/gkarozis/adni/code/test_labels.npy', 'wb') as f:
    np.save(f, test_labels)
with open('/home/gkarozis/adni/code/val_images.npy', 'wb') as f:
    np.save(f, val_images)
with open('/home/gkarozis/adni/code/val_labels.npy', 'wb') as f:
    np.save(f, val_labels)
'''
model = models.Sequential()
model.add(layers.Conv3D(8, (3, 3, 3), activation='relu', input_shape=(110, 160, 88, 1)))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(16, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.build(input_shape = (None, 110, 160, 88, 1))
model.summary()

print('FIRST TIME ' + str(time.time()-start_time))

history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=26, batch_size=4)
this_time = time.time()
print('THE TIME IS '+str(this_time-start_time))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_pred = model.predict_classes(test_images)
print(test_labels)
y_pred = to_categorical(y_pred)
print('y_PREDS', y_pred)
confusion_matrix = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
accuracy = score2(test_labels, y_pred)
precision, recall, fscore, support = score(test_labels, y_pred)
print(confusion_matrix)
df_cm = pd.DataFrame(confusion_matrix, index = ['AD','MCI','CN'],
                  columns = ['AD','MCI','CN'])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
test_loss, test_acc = model.evaluate(test_images,  test_labels)
print(test_loss, test_acc)
print('accuracy: {}'.format(accuracy))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
'''
