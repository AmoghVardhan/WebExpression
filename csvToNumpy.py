import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
for reference

0=Angry
1=Disgust
2=Fear
3=Happy
4=Sad
5=Surprise
6=Neutral
'''


cascade_classifier = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')

def o_h_e(x):
    d = np.zeros(7) #assuming 7 emotions initially
    d[x]=1
    return d

def detectFaceUtil(image):
    if len(image.shape)>2 and image.shape[2] == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image,cv2.CV_LOAD_IMAGE_GRAYSCALE)

    ###preprocessing stage....
    gray_border = np.zeros((150,150),np.uint8)
    gray_border[:,:]=200
    #apply gray border to Image
    gray_border[(75-24):(75+24),(75-24):(75+24)]=image
    #update Image
    image = gray_border
    ###end of preprocessing stage

    ###Face Detection

    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if not len(faces) > 0:
        return None

    #Pass the biggest face found
    #initialize max
    max_area_face = faces[0]

    #fx,fy,fw,fh
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    #crop image
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]


    # Resize image to regular size

    try:
        image = cv2.resize(image, (48,48),
                           interpolation=cv2.INTER_CUBIC) / 255. # to have values only btw 0,1  ### remember that this is done
    except Exception:
        print("[+] Problem during resize")
        return None
    return image



def detectFace(data):
    image = np.fromstring(str(data),dtype=np.uint8,sep=' ').reshape(48,48)
    image = Image.fromarray(image).convert('RGB')
    image = np.array(image)[:, :, ::-1].copy()#RGB to BGR conversion..(opencv uses bgr)# required only incase of using other datasets
    image = detectFaceUtil(image)
    return image

if __name__ == "__main__":
    data = pd.read_csv('./Data/fer2013.csv')
    numImages = data.shape[0]
    labels=[]
    images=[]

    index = 1

    #imageData = data['pixels'].iloc[0]

    for index,row in data.iterrows():
        emotion = o_h_e(row['emotion'])
        image = np.fromstring(str(row['pixels']),dtype=np.uint8,sep=' ').reshape(48,48)
        mpl.image.imsave(('Data/original/'+str(index)+'.png'),image)
        image = detectFace(row['pixels'])
        #color reformat
        #print(image.shape)
        # newImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mpl.image.imsave(('Data/ImageFolder/'+str(index)+'.png'),image)
        if image is not None:
            mpl.image.imsave(('Data/ImageFolder/'+str(index)+'.png'),image)
            labels.append(emotion)
            images.append(image)

        index+=1
        total = numImages
        print("Progress: {}/{} {:.2f}%".format(index,total,index * 100.0/total))

    np.save('./Data/images.npy',images)
    np.save('./Data/labels.npy',labels)
