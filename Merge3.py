import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

from keras.models import load_model

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
    #print(audio)
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
        #print(user)
        return user

    except Exception as e:
        print ("you have not said any thing......... ")
        user=" "
        return user
    return 

intents = json.loads(open('Conversational_Data2.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

i=0
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        if i==0:
            speak("Hello how are you?")
            i=i+1
        arduino.sendData([1])  
        user=takecommand()
        print(user)
        if user!=" ":
            response=chatbot_response(user)
            print(response)
            speak(response)
    else:
        arduino.sendData([0])
        #speak("No one is here")
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)
