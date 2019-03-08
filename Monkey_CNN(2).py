
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


clf = Sequential()


# In[3]:


clf.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = "relu"))


# In[4]:


clf.add(MaxPooling2D(pool_size = (2,2)))


# In[5]:


clf.add(Convolution2D(32,3,3,  activation = "relu"))


# In[6]:


clf.add(MaxPooling2D(pool_size = (2,2)))


# In[7]:


clf.add(Flatten())


# In[8]:


clf.add(Dense(output_dim = 128 , activation = "relu"))


# In[9]:


clf.add(Dense(output_dim = 1, activation = "softmax"))


# In[10]:


clf.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[11]:


from keras.preprocessing.image import ImageDataGenerator


# In[12]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/home/ambarish/Documents/Monkey Dataset/training',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 class_mode='categorical')
        

test_set = test_datagen.flow_from_directory('/home/ambarish/Documents/Monkey Dataset/validation',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            class_mode='categorical')


# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch=(210/5),
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=(90/5))


# In[ ]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset1/single_prediction/p1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Monkey1'
elif result[0][1] == 1:
    prediction = 'Monkey2'
elif result[0][2] == 1:
    prediction = 'Monkey3'
elif result[0][3] == 1:
    prediction = 'Monkey4'
elif result[0][4] == 1:
    prediction = 'Monkey5'
elif result[0][5] == 1:
    prediction = 'Monkey6'
elif result[0][6] == 1:
    prediction = 'Monkey7'
elif result[0][7] == 1:
    prediction = 'Monkey8'
elif result[0][8] == 1:
    prediction = 'Monkey9'
elif result[0][9] == 1:
    prediction = 'Monkey10'
else:
    prediction = 'none'
    
print(prediction)

