
# coding: utf-8

# In[1]:


import os,cv2,glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import keras
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[2]:


PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
train_path= data_path + '/train'
data_dir_list = os.listdir(train_path)

img_rows=128
img_cols=128
num_channel=1

num_epoch=20

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(train_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(train_path + '/'+ dataset + '/'+ img )
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)


# In[3]:


img_data= np.expand_dims(img_data, axis=4)
print (img_data.shape)


# In[4]:


# Define the number of classes
num_classes = 3

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:2283]=0 #2283
labels[2283:5388]=1 #2105
labels[5388:]=2 #245

names = ['benign','malignant','normal']


# In[5]:


# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=10)


# In[6]:


#%%
# Defining the model
input_shape=img_data[0].shape

model = Sequential()

model.add(Convolution2D(64, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"
])


# In[7]:


# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


# In[8]:


from keras import callbacks

filename='model_train_vgg_f(64).csv'
filepath="Best-weights-64_vgg_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]
#callbacks_list = [csv_log]


# In[9]:


# Training
hist = model.fit(X_train, y_train, batch_size=20, epochs=num_epoch, verbose=1, validation_data=(X_val, y_val),callbacks=callbacks_list)


# In[10]:


# visualizing losses and accuracy
get_ipython().run_line_magic('matplotlib', 'inline')

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'bo', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'bo', label='train_acc')
plt.plot(epochs,val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()


# In[11]:


# Evaluating the model with validation images
score = model.evaluate(X_val, y_val, verbose=0)
print('Validation Loss:', score[0])
print('Validation accuracy:', score[1])

val_image = X_val[0:1]
print ('Validation_image.shape',val_image.shape)

print(model.predict(val_image))
print(model.predict_classes(val_image))
print(y_val[0:1])


# In[12]:


PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
test_path= data_path + '/test'
data_dir_list = os.listdir(test_path)

img_rows=128
img_cols=128

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(test_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        test_img=cv2.imread(test_path + '/'+ dataset + '/'+ img )
        test_img=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img_resize=cv2.resize(test_img,(128,128))
        img_data_list.append(test_img_resize)

img_test = np.array(img_data_list)
img_test = img_test.astype('float32')
img_test /= 255
print (img_test.shape)
img_test= np.expand_dims(img_test, axis=4)
print (img_test.shape)


# In[13]:


num_classes = 3

num_of_samples = img_test.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:500]=0 #500
labels[500:1347]=1 #847
labels[1347:]=2 #61

names = ['benign','malignant','normal']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the test
X_test,Y_test = shuffle(img_test,Y, random_state=2)

# Testing
hist = model.fit(X_train, y_train, batch_size=20, epochs=1, verbose=1, validation_data=
(X_test, Y_test),callbacks=callbacks_list)


# In[14]:


#Testing a new image
test_image = cv2.imread('data/test/benign/lgg_01-00034.png',0)
test_image = cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
test_image= np.expand_dims(test_image, axis=3)
test_image= np.expand_dims(test_image, axis=0)
print (test_image.shape)
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))


# In[15]:


# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools
Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class (benign)', 'class 1(malignant)','class 2(normal)']

print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

fig=plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,title='Confusion matrix')

#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
# title='Normalized confusion matrix')
#plt.figure()

plt.show()
fig.savefig("C:/Users/user/Desktop/psm/gui/img/Normalized confusion matrix vgg 64" + '.png')


# In[16]:


#Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model_vgg64.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_vgg64.h5")
print("Loaded model from disk")
model.save('C:/Users/user/Desktop/psm/gui/model/model_vgg64.hdf5')
loaded_model=load_model('C:/Users/user/Desktop/psm/gui/model/model_vgg64.hdf5')

