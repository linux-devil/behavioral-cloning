import csv
import cv2
import numpy as np

lines = []
with open('/Users/harshit.sharma/Downloads/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
images = []
measurements = []
for line in lines:
  '''
  center image
  '''
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = '/Users/harshit.sharma/Downloads/IMG/'+filename
  image = cv2.imread(current_path)
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = imgRGB
  crop_img = imgRGB[40:160,10:310,:]
  image = crop_img
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

  correction = 0.1 # this is a parameter to tune
  steering_left = measurement + correction
  steering_right = measurement - correction
  '''
  left image

  source_path = line[1]
  filename = source_path.split('/')[-1]
  current_path = '/Users/harshit.sharma/Downloads/IMG/'+filename
  image = cv2.imread(current_path)
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = imgRGB
  images.append(image)
  measurements.append(steering_left)
  '''
  '''
  right image

  source_path = line[2]
  filename = source_path.split('/')[-1]
  current_path = '/Users/harshit.sharma/Downloads/IMG/'+filename
  image = cv2.imread(current_path)
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = imgRGB
  images.append(image)
  measurements.append(steering_right)
  '''

augmented_images = []
augmented_measurements = []

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

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5 , input_shape=(120,300,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
model.save('model.h5')
