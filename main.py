import numpy as np
import imageio
import cv2
import socket
import sys
from PIL import Image
from LeNet import LeNet
from AlexNet import AlexNet
from VggNet import VggNet
from customModel import customModel


path1 = ""
path2 = ""
size = 1
obj = None
def trainEvalFunc():
    print('Enter limit for number of epochs:')
    limit = input()
    optEpoch = obj.findOptEpoch(limit)
    model,training,score = obj.train(optEpoch)
    print(training)
    obj.evaluate(model)
    print('Do you want to save the model?(y or n)')
    ans = input()
    if(ans == "y"):
        obj.saveModel(model)

def evalFunc():
    model = obj.loadModel()
    obj.evaluate(model)


dispatch2 = {
    "1": trainEvalFunc,
    "2": evalFunc,
}

def analyzeFunc():
    print('\nChoose operation:')
    print('1.Train Model and evaluate')
    print('2.Evaluate pretrained model')
    print('Enter choice:')
    arg2 = input();
    dispatch2[arg2]()

def readImg():
    cascade_classifier = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')
    print("Which image to load?")
    imgLoad = input()
    image = cv2.imread("testImages/"+imgLoad+".jpeg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
    if(len(faces)==0):
        print("no faces detected")
        exit(0)
    max_area_face = faces[0]

    #fx,fy,fw,fh
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    #crop image
    faces = max_area_face
    print(faces)
    x = faces[0]
    y = faces[1]
    w = faces[2]
    h = faces[3]
    cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.namedWindow('faces found',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('faces found',500,500)
    cv2.imshow("faces found",gray)
    k = cv2.waitKey(0)
    if(k==27):
        cv2.destroyAllWindows()
    gray = gray[y:y+h,x:x+w]

    print("Found {0} faces!".format(len(faces)))
    gray = cv2.resize(gray, (48,48), interpolation = cv2.INTER_AREA)
    return gray


def emotionSwitch(argument):
    switcher = {
        0:"Angry",
        1:"Disgust",
        2:"Fear",
        3:"Happy",
        4:"Sad",
        5:"Surprise",
        6:"Neutral"
    }
    return switcher.get(argument, "Invalid expression")




def emotionFunc():
    image = readImg()
    image = np.array(image)
    testImg = image.reshape(1,size,size,1)
    model = obj.loadModel()
    emotion = model.predict_classes(testImg)[0]
    print("\nEmotion Recognized:")
    print(emotionSwitch(emotion)+"\n")
    # emotionFunc() #comment this line for demo purpose

    emotion = np.asscalar(np.int16(emotion))
    # print(type(emotion))
    # emotion = emotion.tobytes()
    print(emotion)
    HOST, PORT = "192.168.43.2", 27015
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    if emotion is 0:
        s.send(b'0')
    elif emotion is 1:
        s.send(b'1')
    elif emotion is 2:
        s.send(b'2')
    elif emotion is 3:
        s.send(b'3')
    elif emotion is 4:
        s.send(b'4')
    elif emotion is 5:
        s.send(b'5')
    elif emotion is 6:
        s.send(b'6')
dispatch1 = {
    "1":emotionFunc,
    "2":analyzeFunc,
}

def switchFunct(value):

    return{
        '1': lambda :LeNet(train_data,test_data,train_label,test_label),
        '2': lambda :AlexNet(train_data,test_data,train_label,test_label),
        '3': lambda :VggNet(train_data,test_data,train_label,test_label),
        '4': lambda :customModel(train_data,test_data,train_label,test_label)
    }.get(value)()




print("\n\n\n\n----------Emotion Detection using DL----------")
print("Choose Model:")
print("1.LeNet-5")
print("2.AlexNet")
print("3.VggNet(could not train. Please don't choose)")
print("4.Custom Model")
print("Enter choice:")

modelChosen = input()

if modelChosen == "1" or modelChosen == "4":
    path1 = "Data/images.npy"
    path2 = "Data/labels.npy"
    size = 48
    bar = 9813
    barTest = 4205
elif modelChosen=="2" or modelChosen =="3":
    path1 = "Data/images224.npy"
    path2 = "Data/labels224.npy"
    size = 224
    bar = 9813
    barTest =  4205

data = np.load(path1)
label = np.load(path2)

train_data = data[:bar]
test_data = data[bar:]
train_label = label[:bar]
test_label = label[bar:]
train_data = train_data.reshape(bar,size,size,1)
test_data = test_data.reshape(barTest,size,size,1)
obj = switchFunct(modelChosen)
# obj = x(train_data,test_data,train_label,test_label)



print("\nWhat do you intend to do?")
print("1.Detect Emotion")
print("2.Analyze Model Performance")
print("Enter choice:")
arg1 = input()
dispatch1[arg1]()
