import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import model_from_json
data = np.load('Data/images.npy')
label = np.load('Data/labels.npy')




#Basic random model
'''
model = Sequential()
model.add(Dense(10,activation='relu',input_shape=(2304,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_data = data[:9813]
test_data = data[9813:]
train_label = label[:9813]
test_label = label[9813:]
train_data = train_data.reshape(9813,2304)
test_data = test_data.reshape(4205,2304)
model.fit(train_data,train_label,validation_split=0.2,epochs=30)
'''


#LeNet-5
max=0
resI=1
train_data = data[:9813]
test_data = data[9813:]
train_label = label[:9813]
test_label = label[9813:]
train_data = train_data.reshape(9813,48,48,1)
test_data = test_data.reshape(4205,48,48,1)
thresh=[]
for i in range(1,3):
    model = Sequential()
    model.add(Conv2D(6,kernel_size=5,activation='relu',input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2),padding = 'valid'))
    model.add(Conv2D(16,kernel_size=5,activation='relu'))
    model.add(MaxPooling2D(pool_size=2,padding = 'valid'))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dense(7,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    training = model.fit(train_data,train_label,validation_split=0.2,epochs=i)
    score = model.evaluate(test_data,test_label)
    thresh.append(score[1])
    if score[1]>max:
        max = score[1]
        resI = i;

training = model.fit(train_data,train_label,validation_split=0.2,epochs=resI)
score = model.evaluate(test_data,test_label)
print(resI)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
x = np.arange(1,3,1)
y = thresh
plt.plot(x,y)
plt.show()
### optimal epoch chosen is 10 based on graph observation
# model = Sequential()
# model.add(Conv2D(6,kernel_size=5,activation='relu',input_shape=(48,48,1)))
# model.add(MaxPooling2D(pool_size=(2,2),padding = 'valid'))
# model.add(Conv2D(16,kernel_size=5,activation='relu'))
# model.add(MaxPooling2D(pool_size=2,padding = 'valid'))
# model.add(Flatten())
# model.add(Dense(120,activation='relu'))
# model.add(Dense(84,activation='relu'))
# model.add(Dense(7,activation='softmax'))
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# training = model.fit(train_data,train_label,validation_split=0.2,epochs=i)
# score = model.evaluate(test_data,test_label)
model_json = model.to_json()
with open("Models/LeNet-5.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("ModelWeights/LeNet-5__model.h5")
print("saved model to disk")
