
# coding: utf-8

# In[1]:

import matplotlib.image as mpimg
import math
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import random 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam


# In[2]:

flags = tf.app.flags
FLAGS = flags.FLAGS 

flags.DEFINE_float('steering_adjustment', 0.25, "Adjustment angle.")
flags.DEFINE_float('thres', 0.15, "Angle to classify steering.")
flags.DEFINE_integer('epochs', 20, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")


# In[20]:

# Balance the training data for center, left and right-steering data
df = pd.read_csv('data/driving_log.csv')
center = df.center.tolist()
left = df.left.tolist()
right = df.right.tolist()
steering = df.steering.tolist()
plt.hist(steering, bins=50)  # Examine the balance of center, left and right steering
center_copy, steering_copy = center.copy(), steering.copy()
center, steering = shuffle(center, steering)

# X_train, X_valid now contain only center images
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 12)
size = len(center)

idx_c, idx_l, idx_r = [], [], []
for i in range(size):
    if -0.15 < steering[i] < 0.15:
        idx_c.append(i)
    if steering[i] < -0.15:
        idx_l.append(i)
    if steering[i] > 0.15:
        idx_r.append(i)
image_c = [center[i] for i in idx_c]
image_l = [center[i] for i in idx_l] 
image_r = [center[i] for i in idx_r]
steer_c = [steering[i] for i in idx_c]
steer_l = [steering[i] for i in idx_l] 
steer_r = [steering[i] for i in idx_r]
print(len(steer_c), len(steer_l), len(steer_r))


# In[21]:

# Randomly get more images from original data where strong steering left and right steerings are lack of
index_l = random.sample(range(size), len(image_c) - len(image_l))
index_r = random.sample(range(size), len(image_c) - len(image_r))
for i in index_l:
    if steering[i] < -0.15:
        image_l.append(right[i])
        steer_l.append(steering[i] - FLAGS.steering_adjustment)
for i in index_r:
    if steering[i] > 0.15:
        image_r.append(left[i])
        steer_r.append(steering[i] + FLAGS.steering_adjustment)
print(len(steer_c), len(steer_l), len(steer_r))

X_train = image_c + image_l + image_r
y_train = steer_c + steer_l + steer_r


# In[14]:

# disable because too much recovering cause zic zac
'''print(len(y_train))
idx = [i for i in range(len(X_train)) if (y_train[i] < -0.4) | (y_train[i] > 0.4)]
X_train = X_train + [X_train[i] for i in idx]
y_train = y_train + [y_train[i] for i in idx]
y_train = np.float32(y_train)''';


# In[22]:

plt.hist(y_train, bins=50);


# In[23]:

# Generate random brightness function, produce darker transformation 
def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #random_bright = .25+np.random.uniform()
    #hsv[:,:,2] = hsv[:,:,2]*random_bright  # Apply the brightness reduction to the V channel
    
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

# Flip image around vertical axis
def flip(image, angle):
    return cv2.flip(image,1), angle*(-1)

# Crop image to remove the sky and driving deck, resize to 64x64 dimension 
def crop_resize(image):
    cropped = cv2.resize(image[60:140,:,:], (64,64))
    return cropped


# In[8]:

# Training generator: shuffle training data before choosing data. Divide into batches of size as indicated
# Apply random brightness, resize, crop into the chosen sample. Add some small random noise for chosen angle.
'''
def generator_data(batch_size):
    num_samples = len(X_train)
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        X, y = shuffle(X_train, y_train)
        for offset in range(0, num_samples, batch_size):
            batch_image = X[offset:offset+batch_size]
            batch_steer = y[offset:offset+batch_size]
            
            for i in range(len(batch_image)):
                batch_train[i] = crop_resize(random_brightness(mpimg.imread('data/'+batch_image[i].strip())))
                batch_angle[i] = batch_steer[i]*(1+ np.random.uniform(-0.1,0.1))
                #Flip random images
                flip_coin = random.randint(0,1)
                if flip_coin == 1:
                    batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])
            yield batch_train, batch_angle

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_valid(X_valid, y_valid, batch_size):
    num_samples = len(X_valid)
    while True:
        X_valid, y_valid = shuffle(X_valid, y_valid)
        for offset in range(0, num_samples, batch_size):
            batch_image = X_train[offset:offset+batch_size]
            batch_steer = y_train[offset:offset+batch_size]
            
            batch_valid = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
            batch_angle = np.zeros((batch_size,), dtype = np.float32)
            for i in range(len(batch_image)):
                batch_valid[i] = crop_resize(random_brightness(mpimg.imread('data/'+batch_image[i].strip())))
                batch_angle[i] = batch_steer[i]*(1+ np.random.uniform(-0.1,0.1))
            yield batch_valid, batch_angle

data_generator = generator_data(FLAGS.batch_size)
valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)''';


# In[24]:

def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data),1))
            batch_train[i] = crop_resize(random_brightness(mpimg.imread('data/'+data[choice].strip())))
            batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))
            #Flip random images#
            flip_coin = random.randint(0,1)
            if flip_coin == 1:
                batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])

        yield batch_train, batch_angle

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(data,angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(data),1))
            batch_train[i] = crop_resize(mpimg.imread('data/'+data[rand].strip()))
            batch_angle[i] = angle[rand]
        yield batch_train, batch_angle
        
data_generator = generator_data(FLAGS.batch_size)
valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)


# In[25]:

# sequential generator
# Training Architecture: inspired by NVIDIA architecture #
input_shape = (64,64,3)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='valid', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_regularizer = l2(0.001)))
model.add(Dense(1, kernel_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
model.summary()

history_object = model.fit_generator(data_generator, steps_per_epoch = math.ceil(len(X_train)/FLAGS.batch_size), 
    epochs=20, validation_data = valid_generator, validation_steps = math.ceil(len(X_valid)/FLAGS.batch_size))
model.save('model_minh_20.h5')
print("Saved model to disk")


# In[26]:

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:



