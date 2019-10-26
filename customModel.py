import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense, Dropout,Activation
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import model_from_json
from keras import optimizers

class customModel :
    def __init__(self,train_data,test_data,train_label,test_label):
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label


    def train(self,optEpoch):
        model = Sequential()
        model.add(Conv2D(64,kernel_size=5,activation='relu',input_shape=(48,48,1)))
        model.add(MaxPooling2D(pool_size=3,strides = 2,padding = 'valid'))
        model.add(Conv2D(64,kernel_size=5,activation='relu'))
        model.add(MaxPooling2D(pool_size=3,strides = 2,padding = 'valid'))
        model.add(Conv2D(128,kernel_size=4,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(3072,activation='relu'))
        model.add(Dense(7,activation='softmax'))
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.metrics_names)

        training = model.fit(self.train_data,self.train_label,validation_split=0.2,epochs=int(optEpoch))
        score = model.evaluate(self.test_data,self.test_label)
        return model,training,score

    def findOptEpoch(self,limit):
        for i in range(1,int(limit)):
            model,training,score = self.train(i)
        self.plotParam(training)
        print('Enter optEpoch value after visualization:')
        optEpoch = input()
        return optEpoch

    def plotParam(self,training):
        plt.plot(training.history['acc'])
        plt.plot(training.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()

    def saveModel(self,model):
        model_json = model.to_json()
        with open("Models/custom.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("ModelWeights/custom__model.h5")
        print("saved model to disk")

    def loadModel(self):
        json_file = open("Models/custom.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("ModelWeights/custom__model.h5")
        print("loaded model from disk")
        return loaded_model

    def evaluate(self,model):
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        score = model.evaluate(self.test_data,self.test_label)
        print('Test loss:',score[0])
        print('Test accuracy:',score[1])
