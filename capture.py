import numpy as np
from cv2 import cv2 as cv


face_classifier = cv.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    #cropping wajah yang terdeteksi
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

#accessing webcam
cap = cv.VideoCapture(0)
count = 0

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        face = cv.resize(face_extractor(frame), (300, 300))
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        
        # save data wajah pada folder
        file_name_path = './faces/' + str(count) + '.jpg'
        cv.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv.imshow('Merekam data wajah', face)
        count += 1

    else:
        print("Face not found")
        pass

    if cv.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
    
cap.release()
cv.destroyAllWindows()      
print("Data wajah terekam")
