# Surveillance-System
Face Recognition is a computer application that is capable of detecting, tracking, identifying or verifying human faces from an image or video captured using a digital camera. 

------code----------

import cv2
import face_recognition
import numpy as np
from datetime import datetime
import csv

video_capture = cv2.VideoCapture(0)

ani_image = face_recognition.load_image_file("ani.jpg")
ani_encoding=face_recognition.face_encodings(ani_image)[0]

kvee_image = face_recognition.load_image_file("kvee.jpg")
kvee_encoding=face_recognition.face_encodings(kvee_image)[0]

abd_image = face_recognition.load_image_file("abd.jpg")
abd_encoding=face_recognition.face_encodings(abd_image)[0]

known_face_encoding = [
ani_encoding,
kvee_encoding,
abd_encoding
]


known_faces_names = [
"ani",
"kvee",
"abdul"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

#face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
#while 1:
    #ret, img = video_capture.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            
            

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (128,0,0)
                thickness              = 2
                lineType               = 2

                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
            if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

video_capture.release()
cv2.destroyAllWindows()
f.close()
