import tensorflow as tf
import numpy as np
import time
import seaborn as sn
from keras.regularizers import l1,l2
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as score2
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

start_time = time.time()

with open('/home/gkarozis/adni/code/train_images.npy', 'rb') as f:
    train_images = np.load(f)
with open('/home/gkarozis/adni/code/train_labels.npy', 'rb') as f:
    train_labels = np.load(f)
with open('/home/gkarozis/adni/code/test_images.npy', 'rb') as f:
    test_images = np.load(f)
with open('/home/gkarozis/adni/code/test_labels.npy', 'rb') as f:
    test_labels = np.load(f)
with open('/home/gkarozis/adni/code/val_images.npy', 'rb') as f:
    val_images = np.load(f)
with open('/home/gkarozis/adni/code/val_labels.npy', 'rb') as f:
    val_labels = np.load(f)

print(val_images)
print(train_labels.shape, test_labels.shape, val_labels.shape)


model = models.Sequential()
model.add(layers.Conv3D(8, (3, 3, 3), activation='relu', input_shape=(110, 160, 88, 1)))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(16, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu', bias_regularizer=l2(1e-4),activity_regularizer=l2(0.0001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
   '''1.I add Dropout layers in order to make better use of early stopping
      2.More complicated CNN will drive into overfitting (more convolutional layers)
      3.Last Dense layers parameter 3 gives the dimensionality of the output space. (In our problem AD,MCI,CN-3 classes) 
   '''
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.build(input_shape = (None, 110, 160, 88, 1))
model.summary()

print('FIRST TIME ' + str(time.time()-start_time))

history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=60, batch_size=4)
          #The use of 60 epochs is to understand where early stopping begins
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
y_pred = to_categorical(y_pred)
print(y_pred)
z_pred = np.argmax(model.predict(test_images), axis = -1)
confusion_matrix = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
print('Z PREDS', z_pred)
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

this_time = time.time()
print('THE TIME IS '+str(this_time-start_time))
