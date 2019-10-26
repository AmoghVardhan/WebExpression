import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from PIL import Image

class AlexNet:
    def __init__(self,train_data,test_data,train_label,test_label):

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label


    def train(self,optEpoch):

        model = Sequential()
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(224,224,1), kernel_size=(11,11),
         strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

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
        with open("Models/AlexNet1.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("ModelWeights/AlexNet__model1.h5")
        print("saved model to disk")

    def loadModel(self):
        json_file = open("Models/AlexNet.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("ModelWeights/AlexNet__model.h5")
        print("loaded model from disk")
        return loaded_model

    def evaluate(self,model):
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        score = model.evaluate(self.test_data,self.test_label)
        print('Test loss:',score[0])
        print('Test accuracy:',score[1])
