# libraries imported
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
# reading driving log .csv and adding it to training data
with open('/Users/harshit.sharma/Downloads/lap_4/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
for line in lines:
  #center image
  source_path = line[0]
  # left image
  source_path_left = line[1]
  # right image
  source_path_right = line[2]
  # adding center image
  filename = source_path.split('/')[-1]
  current_path = '/Users/harshit.sharma/Downloads/lap_4/IMG/'+filename
  image = cv2.imread(current_path)
  # changing to HLS color space , provides better accuracy
  imgRGB = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  images.append(imgRGB)
  # loading steering angle measurement
  measurement = float(line[3])
  measurements.append(measurement)
  #adding left image and measurement
  filename_left = source_path_left.split('/')[-1]
  current_path_left = '/Users/harshit.sharma/Downloads/lap_4/IMG/'+filename_left
  image_left = cv2.imread(current_path_left)
  imgRGB_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2HLS)
  images.append(imgRGB_left)
  # correction factor of 0.2
  measurement = float(line[3]) + 0.2
  measurements.append(measurement)
  # adding right image and measurement
  filename_right = source_path_right.split('/')[-1]
  current_path_right = '/Users/harshit.sharma/Downloads/lap_4/IMG/'+filename_right
  image_right = cv2.imread(current_path_right)
  imgRGB_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2HLS)
  images.append(imgRGB_right)
  # correction factor of 0.2
  measurement = float(line[3]) - 0.2
  measurements.append(measurement)

indices = [i for i, x in enumerate(measurements) if x == float(0)]
shuffle(indices)
print(len(indices))
# removing 80 percent of training set with 0 steering angle
perc_75 = 0.80*len(indices)
to_remove = int(perc_75)
indices = indices[:to_remove]

augmented_images = []
augmented_measurements = []

final_images = []
final_measurements = []
# following code is generating training data
for i in range(len(measurements)):
    if i not in indices:
        final_images.append(images[i])
        final_measurements.append(measurements[i])

measurements = final_measurements
images = final_images
# augmenting training data with flip function
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
# cropping data below to remove noise such as sky and scenery around which does not influence steering angle
model.add(Cropping2D(crop_shape, input_shape=input_shape, name="Crop"))
# Normalize input.
model.add(Lambda(lambda x:x/255.0 - 0.5 ))
# conv2d layer added
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
# adding optimizer below
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
#saving_model
model.save('model_new.h5')
