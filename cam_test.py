'''
USAGE:
python cam_test.py 
'''
from PIL import Image
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import argparse
import torch.nn.functional as F
import time
import pickle
from torchvision import models
from feature_extraction import extract_features_hog
import mediapipe as mp
# load label binarizer
# lb = joblib.load('lb.pkl')

with open('model.pickle', 'rb') as f:
    hmm_model = pickle.load(f)
    
print(hmm_model)
print('Model loaded')


dict = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",
        17:"H",18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",
        33:"X",34:"Y",35:"Z"}

def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224,224))
    return hand

def predict_gesture(img):
    features = extract_features_hog(image)
    print(features.shape)
    
    n_timesteps = features.shape[0]
    n_features = 1
    seq = features.reshape(n_timesteps, n_features)
    score = hmm_model.score(seq)    
    predicted_sequence = hmm_model.predict(seq)
    
    predicted_probability = np.max(predicted_sequence)
    probabilities_formatted = "{:.2f}%".format(predicted_probability )

    predicted_class = np.argmax(predicted_sequence)
    print("The predicted class label is:", dict[predicted_class])
    print("\n\n")
    print(f'The predicted image corresponds to "{dict[predicted_class]}" with {probabilities_formatted} probability.')

    return dict[predicted_class], predicted_probability


cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter('./outputs/asl.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    # get the hand area on the video capture screen
    cv2.rectangle(frame, (100, 100), (324, 324), (20,34,255), 2)
    hand = hand_area(frame)
  
    image = hand
    
    predicted_class, predicted_probability = predict_gesture(image)
    cv2.putText(frame, predicted_class[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    out.write(frame)

    time.sleep(0.09)

    # press `q` to exit
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
    

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()