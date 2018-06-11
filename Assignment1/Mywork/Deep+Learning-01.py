
# coding: utf-8

# In[ ]:


# Shenghua Du


# # 1 Getting Started

# In[2]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import keras.utils

def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata   #return a list 

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = 1.0    
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V, noise)


def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]


im = generate_a_rectangle(10, True)
plt.imshow(im.reshape(72,72), cmap='gray')

im = generate_a_disk(10)
plt.imshow(im.reshape(72,72), cmap='gray')

[im, v] = generate_a_triangle(20, False)
plt.imshow(im.reshape(72,72), cmap='gray')


def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
             pass
 #           print(i)     I commened this out to save print space
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1: 
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(300, 20, True)
    Y_test = np_utils.to_categorical(Y_test, 3) 
    return [X_test, Y_test]

def generate_dataset_regression(nb_samples, noise=0.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples, 6])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
             pass
#            print(i)    I commened this out to save print space
        [X[i], Y[i]] = generate_a_triangle(noise, True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

import matplotlib.patches as patches

def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((72,72))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()

def generate_test_set_regression():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(300, 20)
    Y_test = np_utils.to_categorical(Y_test, 3) 
    return [X_test, Y_test]


# # 2 Simple Classification 

# In[4]:


# import library
from keras.models import Sequential 
from keras import layers
from keras.layers import Dense, Activation,Flatten,BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input,UpSampling2D,merge,Conv2D,Cropping2D,concatenate
from keras.models import Model


# In[5]:


# generate the train set, and transfor the y_train to a category from an int
from keras import utils as np_utils
[X_train, Y_train] = generate_dataset_classification(300, 20)
Y_train = np_utils.to_categorical(Y_train)


# In[6]:


model = Sequential()
n_cols = X_train[0].shape

#one single layer, and use softmax  
model.add(Dense(units=3,activation='softmax',input_shape=n_cols))

# use the Adam optimizor 
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

# fit the model, got training acc = 1 
model.fit(X_train, Y_train, epochs=20, batch_size=32)


# In[7]:


X_test = generate_a_disk()
X_test = X_test.reshape(1, X_test.shape[0])
model.predict(X_test)


# #3 Visualization of the Solution

# In[8]:


# get the maxtrix of the classifie, and visualize the 3 columns 
weights = model.get_weights()
fig, ax = plt.subplots(figsize=(5, 5))
I = weights[0][:,0].reshape((72,72))
ax.imshow(I,cmap='hot')


# In[9]:


weights = model.get_weights()
fig, ax = plt.subplots(figsize=(5, 5))
I = weights[0][:,1].reshape((72,72))
ax.imshow(I,cmap='hot')


# In[10]:


weights = model.get_weights()
fig, ax = plt.subplots(figsize=(5, 5))
I = weights[0][:,2].reshape((72,72))
ax.imshow(I,cmap='hot')


# # 4 A More Difficult Classification Problem

# In[11]:


#generate the new training set, I used 2000 samples in this problem
[X_train, Y_train] = generate_dataset_classification(2000, 20, True)

# reshape x and y 
Y_train = np_utils.to_categorical (Y_train,3)
X_train = X_train.reshape(X_train.shape[0],1,72,72)


# In[12]:



model_2 = Sequential()
# input convolution layer
model_2.add(Convolution2D(16,(5,5),activation = 'relu',input_shape=(1,72,72),data_format="channels_first"))
#maxpooling layer
model_2.add(MaxPooling2D(pool_size=(2, 2)))
#Flatten
model_2.add(Flatten())
#full_connected layer 
model_2.add(Dense(units=30, activation = 'relu' ))
#ouput layer 
model_2.add(Dense(units=3,activation='softmax'))


# In[13]:


#compile the model_2
model_2.compile(loss='categorical_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])

# fit the model_2, get training acc around 0.99 
model_2.fit(X_train, Y_train, epochs=32, batch_size=32)


# In[14]:


# model.evaluate(X_test, Y_test), got acc around 0.93
[X_test, Y_test] = generate_test_set_classification()
X_test = X_test.reshape(X_test.shape[0],1,72,72)
model_2.evaluate(X_test, Y_test)


# # 5 A Regression Problem

# In[15]:


# generate training set, and reshape the data 
[X_train, Y_train] = generate_dataset_regression(1000, 20)
X_train = X_train.reshape(-1,1,72,72)
Y_train = Y_train.reshape(1000,6)


# In[16]:


visualize_prediction(X_train[1], Y_train[1])


# In[17]:


# normalize the Y set
# Firstly, I calcualted the distance between each vertice and the point (0,0)
# Then I resorted the Y set by the distance 

from itertools import chain

def norm (Y):
  new_Y =[]
  length = Y.shape[0]
  for i in range(length):
    list_y = list(Y[i])
    it = iter(list_y)
    b = list(zip(it,it))
    y = sorted((x for x in b), key = lambda x : x[0]**2+x[1]**2)
    new_Y.append(list(chain.from_iterable(y)))
  return (np.array(new_Y).reshape(length,6))

  


# In[18]:


model_3 = Sequential()


# input convolution layer
model_3.add(Convolution2D(16,(5,5),activation = 'relu',input_shape=(1,72,72),data_format="channels_first",padding='same'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Convolution2D(32,(5,5),activation = 'relu',padding = 'same'))
model_3.add(MaxPooling2D(pool_size = (2,2)))

model_3.add(Convolution2D(64,(5,5),activation = 'relu',padding = 'same'))
model_3.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
model_3.add(Flatten())

model_3.add(Dense(units=50,activation = 'relu'))

model_3.add(Dense(units=6,activation='softmax'))


# In[19]:


#compile the model_3
model_3.compile(loss='mean_squared_logarithmic_error',optimizer = 'Adam',metrics = ['accuracy'])

# fit the model_3, got accuracy around 0.83
model_3.fit(X_train, norm(Y_train), epochs=30, batch_size=30)


# In[20]:


# There is a problem with the generate_dataset_regression function, which outputs y as category. 
#Therefore ,I use the generate_dataset_regression to generate the test set 

# the evaluation accuracy is around 0.75

[X_test, Y_test] = generate_dataset_regression(100,20)
X_test = X_test.reshape(-1,1,72,72)
model_3.evaluate(X_test, norm(Y_test))


# # Bonus Question 

# In[21]:


# change the generate_a_* functions, each function will return two graph, one with noise = 50 and another with noise =0 


def generate_pari_rectangle(noise=0, free_location=False):
    figsize = 1.0    
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V,noise=50),generate_a_drawing(figsize,U,V,noise=0)

def generate_pair_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise=50),generate_a_drawing(figsize, U, V, noise=0)

def generate_pair_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise=50),generate_a_drawing(figsize, U, V, noise=0)
    return imdata



# In[22]:


# use the generate_pair_triangle to show a pair smaple
[im,im_2] = generate_pair_triangle()

plt.imshow(im_2.reshape(72,72), cmap='gray')



# In[23]:


plt.imshow(im.reshape(72,72), cmap='gray')


# In[24]:


# redefine this function, to generate a X_train set ( graph with noise), and a Y_train set (same graph without noise)

def generate_dataset_denosing(nb_samples, free_location=False):
    X_train=[]
    Y_train=[]
    print('Creating data:')
    for i in range(nb_samples):
        category = np.random.randint(0,2)
        if category == 0:
          [ima_1,ima_2]=generate_pari_rectangle()
          a = (ima_1+50)/(255+100)
          b= ima_2/255
        elif category == 1: 
          [ima_1,ima_2] = generate_pair_disk()
          a = (ima_1+50)/(255+100)
          b = ima_2/255
            
        else:
          [ima_1,ima_2] = generate_pair_triangle()
          a = (ima_1+50)/(255+100)
          b = ima_2/255
        X_train.append(a)
        Y_train.append(b)
    print ('Data Created')
    return X_train,Y_train


# In[25]:


# generate the training data, and reshape the data 
[X_train,Y_train] = generate_dataset_denosing(300)
X_train = np.reshape(X_train,(300,72,72,1))
Y_train = np.reshape(Y_train,(300,72,72,1))


# In[26]:


# build the model 

input_img = Input(shape=(72,72,1)) 


# the encode part 
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(input_img) 
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


# the encode part 
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x) 
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, (1, 1), activation='sigmoid', padding='same')(x)


model_4 = Model(input_img, decoded)
model_4.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

          


# In[27]:


# fit the model, get acc = 0.98
model_4.fit(X_train,Y_train,epochs=20, batch_size=32)


# In[28]:


# use the evaluate function, and get accuracy 0.98 
[X_test,Y_test] = generate_dataset_denosing(100)
X_test = np.reshape(X_test,(100,72,72,1))
Y_test = np.reshape(Y_test,(100,72,72,1))
model_4.evaluate(X_test, Y_test)


# In[29]:


plt.imshow(X_test[0].reshape(72,72),cmap='gray')


# In[30]:


# plot an output sample 
a=model_4.predict(X_test[0].reshape(1,72,72,1))
plt.imshow(a.reshape(72,72),cmap='gray')

