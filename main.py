import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import dlib
import time 
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance as dist
from pygame import mixer
from imutils import face_utils

mixer.init()
sound = mixer.Sound('alarm.mp3')


labels = os.listdir("./train")

import matplotlib.pyplot as plt
plt.imshow(plt.imread("./train/Closed/_0.jpg"))
a = plt.imread("./train/yawn/10.jpg")

plt.imshow(plt.imread("./train/yawn/10.jpg"))

# Function to crop the face and detect if yawning or not
def face_for_yawn(direc="./train", face_cas_path="./haarcascade/haarcascade_frontalface_default.xml" ):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


yawn_no_yawn = face_for_yawn()


# Function to detect open or closed eyes
def get_data(dir_path="./train/", face_cas="./haarcascade/haarcascade_frontalface_default.xml", eye_cas="../input/prediction-images/haarcascade.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num += 2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

data_train = get_data()


def append_data():
#     total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)

new_data = append_data()

X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

# Reshape the array
X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

# Label binarizer
from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)

y = np.array(y)

# Train test split
from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

print("x test")
print(len(X_test))


# Data augmentation
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# Model
model = Sequential()

model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()



face_cascade = cv2.CascadeClassifier(r'C:\Users\user\Desktop\Drowsiness_detection_system\haarcascade\haarcascade_frontalface_default.xml')

lefteye_cascade = cv2.CascadeClassifier(r'C:\Users\user\Desktop\Drowsiness_detection_system\haarcascade\haarcascade_lefteye_2splits.xml')
righteye_cascade = cv2.CascadeClassifier(r'C:\Users\user\Desktop\Drowsiness_detection_system\haarcascade\haarcascade_righteye_2splits.xml')


cap = cv2.VideoCapture(0)
label = ["Open", "Closed", 'Yawning', "Not yawning"];

def cal_yawn(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
  
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
  
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
  
    distance = dist.euclidean(top_mean,low_mean)
    return distance
  


# cam = cv2.VideoCapture('http://192.168.1.50:4747/video')
  


#--------Variables-------#
yawn_thresh = 35
ptime = 0
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
lbl=['Close','Open', "Yawning", "Not Yawning"]
rect = 0


face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  
while 1:
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #---------FPS------------#    
    ctime = time.time() 
    fps= int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(frame,f'FPS:{fps}',(frame.shape[1]-120,frame.shape[0]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)
  

    faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = lefteye_cascade.detectMultiScale(gray)
    right_eye =  righteye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 1 )
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    
    # Yawning detection
    for face in faces:
        
        #----------Detect Landmarks-----------#
        shapes = landmark_model(gray, rect) 
        shape = face_utils.shape_to_np(shapes)
    
        #-------Detecting/Marking the lower and upper lip--------#
        lip = shape[48:60]
        cv2.drawContours(frame,[lip],-1,(0, 165, 255),thickness=3)
    
        #-------Calculating the lip distance-----#
        lip_dist = cal_yawn(shape)
        # print(lip_dist)
        if lip_dist > yawn_thresh : 
            cv2.putText(frame, f'User Yawning!',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)  

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        r_eye=frame[y:y+h,x:x+w]
        count += 1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        elif(rpred[0]==0):
            lbl='Closed'
        print(lbl)
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1.5 )
        l_eye=frame[y:y+h,x:x+w]
        count += 1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if(lpred[0]==1):
            lbl='Open'   
        elif(lpred[0]==0):
            lbl='Closed'
        print(lbl)
        break


   
  


    if(rpred[0]==0 and lpred[0]==0):
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1,(255,255,255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score -= 5
        cv2.putText(frame,"Open",(10, height-20), font, 1,(255,255,255), 1, cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:' + str(score), (100, height - 20), font, 1,(255,255,255), 1, cv2.LINE_AA)
    if(score==15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            continue

        except:  # isplaying = False
            pass
        if(thicc < 16):
            thicc += 2
        else:
            thicc -= 2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
