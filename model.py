import csv
import cv2
import numpy as np
from random import shuffle
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Lambda, AveragePooling2D
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.normalization import BatchNormalization


images = []
measurements = []

lines = []
with open('/Users/harshit.sharma/Downloads/lap_4/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
for line in lines:
  #center image
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = '/Users/harshit.sharma/Downloads/lap_4/IMG/'+filename
  image = cv2.imread(current_path)
  #imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imgRGB = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  image = imgRGB
  crop_img = imgRGB[40:160,10:310,:]
  image = crop_img
  images.append(imgRGB)
  measurement = float(line[3])
  measurements.append(measurement)

indices = [i for i, x in enumerate(measurements) if x == float(0)]
shuffle(indices)
print(len(indices))
perc_75 = 0.80*len(indices)
to_remove = int(perc_75)
indices = indices[:to_remove]

augmented_images = []
augmented_measurements = []

final_images = []
final_measurements = []

for i in range(len(measurements)):
    if i not in indices:
        final_images.append(images[i])
        final_measurements.append(measurements[i])

measurements = final_measurements
images = final_images

for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
input_shape = [160, 320, 3]
#theta.crop_shape = ((80,20),(1,1))
crop_shape= ((80,20),(1,1))
#input_shape=(120,300,3)
model = Sequential()
#model.add(Lambda(lambda x:x/255.0 - 0.5 , input_shape=(120,300,3)))
model.add(Cropping2D(crop_shape, input_shape=input_shape, name="Crop"))
model.add(Lambda(lambda x:x/255.0 - 0.5 ))
# Normalize input.
#model.add(BatchNormalization(axis=1, name="Normalize"))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.05, name="Dropout"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
model.save('modelwithaug_lap4.h5')
