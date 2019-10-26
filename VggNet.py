import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D,Convolution2D
from keras.layers import Flatten
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from PIL import Image

class VggNet:
    def __init__(self,train_data,test_data,train_label,test_label):

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label


    def train(self,optEpoch):

        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # Output Layer
        model.add(Dense(7))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        training = model.fit(self.train_data,self.train_label,validation_split=0.2,epochs=int(optEpoch))
        score = model.evaluate(self.test_data,self.test_label)
        return model,training,score

    def findOptEpoch(self,limit):
        thresh=[]
        for i in range(1,int(limit)):
            model,training,score = self.train(i)
            thresh.append(score[1])
        print('Enter optEpoch value after visualization:')
        optEpoch = input()
        return optEpoch

    def plotParam(self,limit,thresh):
        x = np.arange(1,int(limit),1)
        y = thresh
        print(set(zip(x,y)))

        plt.plot(x,y)
        plt.show()

    def saveModel(self,model):
        model_json = model.to_json()
        with open("Models/VggNet.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("ModelWeights/VggNet__model.h5")
        print("saved model to disk")

    def loadModel(self):
        json_file = open("Models/VggNet.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("ModelWeights/VggNet__model.h5")
        print("loaded model from disk")
        return loaded_model

    def evaluate(self,model):
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        score = model.evaluate(self.test_data,self.test_label)
        print('Test loss:',score[0])
        print('Test accuracy:',score[1])
