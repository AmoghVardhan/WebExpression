import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import model_from_json

class LeNet:
    def __init__(self,train_data,test_data,train_label,test_label):
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label


    def train(self,optEpoch):
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
        print(model.metrics_names)

        training = model.fit(self.train_data,self.train_label,validation_split=0.2,epochs=int(optEpoch))
        score = model.evaluate(self.test_data,self.test_label)
        return model,training,score

    def findOptEpoch(self,limit):
        thresh=[]
        for i in range(1,int(limit)):
            model,training,score = self.train(i)
            thresh.append(score[1])
        self.plotParam(limit,thresh)
        print('Enter optEpoch value after visualization:')
        optEpoch = input()
        return optEpoch

    def plotParam(self,limit,thresh):
        x = np.arange(1,int(limit),1)
        y = thresh
        plt.plot(x,y)
        plt.show()

    def saveModel(self,model):
        model_json = model.to_json()
        with open("Models/LeNet-5.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("ModelWeights/LeNet-5__model.h5")
        print("saved model to disk")

    def loadModel(self):
        json_file = open("Models/LeNet-5.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("ModelWeights/LeNet-5__model.h5")
        print("loaded model from disk")
        return loaded_model

    def evaluate(self,model):
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        score = model.evaluate(self.test_data,self.test_label)
        print('Test loss:',score[0])
        print('Test accuracy:',score[1])
