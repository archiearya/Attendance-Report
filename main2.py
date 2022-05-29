from cProfile import label
from email.mime import image
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter
window =  tkinter.Tk()
window.title("PROXY DETECTED")
#window.geometry('350x200')
# from PIL import ImageGrab

#images and studentNames store images and names of students respectively
path = 'Training_images'
images = []
studentNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    studentNames.append(os.path.splitext(cl)[0])
print(studentNames)

# convert from bgr to rgb 
# face encoding func of the face recog lib 
# every encoding appended in encodingList var
def findEncodings(images):
    encodingList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

# 
def attendanceMark(name):
    with open('AttendanceClass.csv', 'r+') as f:
        dataList = f.readlines()


        studentList = []
        for line in dataList:
            entry = line.split(',')
            studentList.append(entry[0])
            if name not in studentList:
                dtString = datetime.now().strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                break


def runner():
    encodingListKnown = findEncodings(images)
    print('Encoding is Complete')

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurrentFrame = face_recognition.face_locations(imgS)
        encodesCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, faceCurrentFrame):
            matched = face_recognition.compare_faces(encodingListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)
    
            matchIndex = np.argmin(faceDis)

            if matched[matchIndex]:
                name = studentNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                attendanceMark(name)
            else:
                
                icon = tkinter.PhotoImage(file = 'D:\Attendance-Report\PSWarning.png')
                label = tkinter.Label(window,image=icon)
                label.pack()
                bt = tkinter.Button(window, text = "TRY AGAIN", bg="orange", fg ="red",command = runner)
                #bt.grid(column = 1, row=0)
                bt.pack()
                window.mainloop()
                print('Proxy was detected, please try again!')
                
               
                
    

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
                
runner()
