# trafficsignlanedet
# Mounting Google Drive

from google.colab import drive drive.mount('/content/gdrive', force_remount=True)

# Importing Libraries

import numpy as np import random import os
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout

# Reading and Pre-processing Images

images = [] labels = [] classes = 43
current_path = '/content/gdrive/My Drive/GTSRB/Final_Training/Images/' for i in range(classes):
path = os.path.join(current_path, str(str(i).zfill(5))) img_folder = os.listdir(path)
for j in img_folder: try:
image = cv.imread(str(path+'/'+j)) image = cv.resize(image, (32, 32))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) image = np.array(image)
images.append(image) label = np.zeros(classes) label[i] = 1.0 labels.append(label)
except:
 
pass

images = np.array(images) images = images/255 labels = np.array(labels)
print('Images shape:', images.shape) print('Labels shape:', labels.shape)

# Splitting the Dataset into Train,Test and Validation subsets
X = images.astype(np.float32) y = labels.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
X_val=X_train[:5500] y_val=y_train[:5500]
print('X_train shape:', X_train.shape) print('y_train shape:', y_train.shape) print('X_test shape:', X_test.shape) print('y_test shape:', y_test.shape)

# Random 30 images from Dataset
plt.figure(figsize=(20, 20))
start_index = 0
for i in range(30): plt.subplot(6, 5, i+1) plt.grid(True) plt.axis('on')
plt.xticks([])
plt.yticks([])
label = np.argmax(y_train[start_index+i]) plt.xlabel('CLASS={}'.format(label)) plt.imshow(X_train[start_index+i])
plt.show()

# Building the model
model = Sequential([
Rescaling(1, input_shape=(32, 32, 1)),
Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
 
AveragePooling2D(pool_size=(2, 2)),
Conv2D(filters=16, kernel_size=(5, 5), activation='relu'), AveragePooling2D(pool_size=(2, 2)),
Conv2D(filters=120, kernel_size=(5, 5), activation='relu'), Dropout(0.2),
Flatten(),
Dense(units=120, activation='relu'), Dense(units=43, activation='softmax')
])
# Model Compilation
model.compile( optimizer='adam',
loss='categorical_crossentropy', metrics=['accuracy']
)

# Model Architecture
model.summary()
history = model.fit(X_train, y_train, epochs=50,validation_data=(X_val, y_val))
Validation Accuracy and loss
val_loss, val_acc = model.evaluate(X_train[:5500], y_train[:5500], verbose=2) print('\nValidation accuracy:', val_acc)
print('\nValidation loss:', val_loss) plt.figure(0)
plt.plot(history.history['accuracy'], label='accuracy') plt.plot(history.history['val_accuracy'], label = 'val_accuracy') plt.xlabel('Epoch')
plt.ylabel('Accuracy') plt.ylim([0.8, 1]) plt.legend(loc='lower right') plt.figure(1)
plt.plot(history.history['loss'], label='loss') plt.plot(history.history['val_loss'], label = 'val_loss') plt.xlabel('Epoch')
plt.ylabel('Loss') plt.ylim([0, 0.2]) plt.legend(loc='lower right')

# Testing Accuracy and loss
 
test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2) print('\n Test accuracy:',test_acc)
print('\n Test loss:', test_loss)

Saving the model

model.save('/content/gdrive/My Drive/keras_model/Trafic_signs_model.h5')
