import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Activation, Dense, Flatten ,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


#loads the data 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#flattens the 1d vector length to 784 

#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = Sequential()

from ann_visualizer.visualize import ann_viz
ann_viz(model)

model = Sequential()
 
model.add(Flatten())
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
 
sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


history = model.fit(x_train, y_train,
                batch_size= 128,
                epochs= 50,
                verbose=1,
                validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

history_dict = history.history

print(history_dict.keys()) 

plt.plot(range(50), history_dict['loss'], label='Loss') 
plt.plot(range(50), history_dict['categorical_accuracy'], label='Accuracy') 
plt.plot(range(50), history_dict['val_categorical_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.legend()
plt.show()
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.show()