from cv2 import cv2 as cv
import numpy as np

def recognizer():
    face_classifier = cv.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
    def face_detector(img, size=0.5):
        
        # Convert image to grayscale
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 4)
        if faces is ():
            return img, []
        
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv.resize(roi, (200, 200))
        return img, roi


    # Open Webcam
    cap = cv.VideoCapture(0)
    model = cv.face.LBPHFaceRecognizer_create() 
    model.read('faces data/faces_data.yml')

    while True:

        ret, frame = cap.read()
        
        image, face = face_detector(frame)
        
        try:
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

            # Pass face to prediction model
            # result from tupplr
            results = model.predict(face)
            
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% match'
                
            cv.putText(image, display_string, (250, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            
            if confidence > 85:
                cv.putText(image, "Unlocked", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv.imshow('Face Recognition', image )
                
            else:
                cv.putText(image, "Locked", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv.imshow('Face Recognition', image )

        except:
            cv.putText(image, "No Face Found", (200, 100) , cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv.putText(image, "Locked", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv.imshow('Face Recognition', image )
            pass
            
        if cv.waitKey(1) == 27: #esc to terminate
            break
            
    cap.release()
    cv.destroyAllWindows()
