import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject

cap = cv2.VideoCapture(0)
detector = FaceDetector()
arduino = SerialObject()
import pyttsx3
import speech_recognition as sr
engine = pyttsx3.init('sapi5')
voices = engine.getProperty("voices")
engine.setProperty("Voices", voices[0].id)

def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()

def takecommand():
    r = sr.Recognizer()
    user=""
    with sr.Microphone()as source:
        print("Listening")
        r.pause_threshold = 1
        audio =r.listen(source,timeout=1,phrase_time_limit=5)
    try:
        print("Recognizing")
        user =r.recognize_google(audio,language= 'en-pk')
        print(user)
        return user

    except Exception as e:
        speak ("you have not said any thing......... ")
        return
    return user


while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        arduino.sendData([1])
        speak("Hellow how are you?")
        takecommand()
        
    else:
        arduino.sendData([0])
        speak("No one is here")
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)
