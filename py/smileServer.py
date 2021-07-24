from IPython.core.display import _JPEG
import matplotlib.pyplot as plt
from pylab import *
from sklearn import datasets
import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from scipy.stats import sem
import numpy as np
import cv2
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
import os

#################################################################################################################################
# Load the faces and classify using buttons
#################################################################################################################################
number_of_faces = 201
faces = datasets.fetch_olivetti_faces()
faces.keys()

for i in range(number_of_faces):
    face = faces.images[i]
    subplot(1, number_of_faces, i + 1)
    imshow(face.reshape((64, 64)), cmap='gray')
    axis('off')
#show()

class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0
        
    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                print(self.index)
                self.index += 1
            return self.index
    
    def record_result(self, smile=True):
        self.results[str(self.index)] = smile

trainer = Trainer()

button_smile = widgets.Button(description='smile')
button_no_smile = widgets.Button(description='sad face')
out = widgets.Output()

def display_face(face):
    clear_output()
    imshow(face, cmap='gray')
    axis('off')

def update_smile(b):
    with out:
        clear_output()
        trainer.record_result(smile=True)
        trainer.increment_face()
        display_face(trainer.imgs[trainer.index])

def update_no_smile(b):
    with out:
        clear_output()
        trainer.record_result(smile=False)
        trainer.increment_face()
        display_face(trainer.imgs[trainer.index])

button_no_smile.on_click(update_no_smile)
button_smile.on_click(update_smile)

display(button_smile)
display(button_no_smile)
display_face(trainer.imgs[trainer.index])

buttons = widgets.HBox([button_smile, button_no_smile])
widgets.VBox([buttons,out])

#################################################################################################################################
# Save the classified faces in an xml file
#################################################################################################################################
#with open('results.xml', 'w') as f:
#    json.dump(trainer.results, f)

#################################################################################################################################
# Load the classified faces from the xml file
#################################################################################################################################
results = json.load(open('results.xml'))
trainer.results = results

#################################################################################################################################
# Displays faces based on classification
#################################################################################################################################
# yes, no = (sum([trainer.results[x] == True for x in trainer.results]), 
#              sum([trainer.results[x] == False for x in trainer.results]))
# bar([0, 1], [no, yes])
# ylim(0, max(yes, no))
# xticks([0.4, 1.4], ['no smile', 'smile'])

# smiling_indices = [int(i) for i in results if results[i] == True]
# not_smiling_indices = [int(i) for i in results if results[i] == False]

# fig = plt.figure(figsize=(12, 12))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(len(smiling_indices)):
#      # plot the images in a matrix of 20x20
#      p = fig.add_subplot(20, 20, i + 1)
#      p.imshow(faces.images[smiling_indices[i]], cmap=plt.cm.bone)
    
#      # label the image with the target value
#      p.text(0, 14, "smiling")
#      p.text(0, 60, str(i))
#      p.axis('off')
    
# fig = plt.figure(figsize=(12, 12))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(len(not_smiling_indices)):
#     # plot the images in a matrix of 20x20
#     p = fig.add_subplot(20, 20, i + 1)
#     p.imshow(faces.images[not_smiling_indices[i]], cmap=plt.cm.bone)

#     # label the image with the target value
#     p.text(0, 14, "not smiling")
#     p.text(0, 60, str(i))
#     p.axis('off')

#################################################################################################################################
# Trains model and prints mean accuracy
#################################################################################################################################

svc_1 = svm.SVC(kernel='linear')
indices = [i for i in trainer.results]
#print(indices)
data = faces.data[0:201]
#print(data)
#data = faces.data[int(i) for i in [i for i in trainer.results]]
target = [trainer.results[i] for i in trainer.results]
target = array(target).astype(int32)
#print(target)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data, target, test_size=0.25, random_state=0)
def evaluate_cross_validation(clf, X, y):
    # create a k-fold cross validation iterator
    cv = model_selection.KFold(len(y), shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = model_selection.cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))
evaluate_cross_validation(svc_1, X_train, y_train)

#################################################################################################################################
# Extra information about training model - classification report and confusion matrix
#################################################################################################################################

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    
train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

#################################################################################################################################
# Build simple UI to check if model works
#################################################################################################################################

# random_image_button = widgets.Button(description="New image!")

# def display_face_and_prediction(b):
#      with out:
#         clear_output()
#         index = randint(0, 201)
#         face = faces.images[index]
#         display_face(face)
#         #print(face.shape)
#         #print("this person is smiling: {0}".format(svc_1.predict(np.reshape(faces.data[index, :], (64,64)))==1))
#         prediction = (svc_1.predict(faces.data[0:201]) == 1)
#         #print(prediction)
#         print("this person is smiling: {0}".format(prediction[index]))


# random_image_button.on_click(display_face_and_prediction)
# display(random_image_button)
# display_face_and_prediction(0)

# buttons = widgets.HBox([random_image_button])
# widgets.VBox([buttons,out])

#################################################################################################################################
# Use OpenCV to take an image, draw bounding box, and return a result
#################################################################################################################################

input_face = cv2.imread("images/face1.jpg")
cascPath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
    #faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
    #df = faceCascade.detectMultiScale(gray)
detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
)

#################################################################################################################################
# Changes image based on coefficients given so that it matches the 64 by 64 pixel size of original dataset and checks if
# model is able to predict the image as smiling or not smiling
#################################################################################################################################

ax = gca()
ax.imshow(gray, cmap='gray')
for (x, y, w, h) in detected_faces:
    ax.add_artist(Rectangle((x, y), w, h, fill=False, lw=5, color='blue'))

    original_extracted_face = gray[y:y+h, x:x+w]
    horizontal_offset = int(0.15 * w)
    vertical_offset = int(0.2 * h)
    extracted_face = gray[y+vertical_offset:y+h, 
                        x+horizontal_offset:x-horizontal_offset+w]

    subplot(121)
    imshow(original_extracted_face, cmap='gray')
    subplot(122)
    imshow(extracted_face, cmap='gray')

    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 
                                                   64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(float32)
    new_extracted_face /= float(new_extracted_face.max())
#    display_face(new_extracted_face[:, :])

    p = svc_1.predict(new_extracted_face.ravel().reshape(1, -1))
        #plt.imshow(new_extracted_face.ravel())
    print(p)
    if p == 1:
        print("this person is smiling")
    else:
        print("this person is not smiling")

#################################################################################################################################
# Uses the same principle of getting extracted features but for my own face (I gave images of me smiling and not smiling).
#################################################################################################################################

def detect_face(frame):
    cascPath = "haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, detected_faces

def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y+vertical_offset:y+h, 
                      x+horizontal_offset:x-horizontal_offset+w]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 
                                           64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

def predict_face_is_smiling(extracted_face):
    return svc_1.predict(extracted_face.ravel().reshape(1, -1))

# point to one smiling and one non-similing face of yours
smiling_face = "images/face2.jpg"
non_smiling_face = "images/face3.jpg"

subplot(121)
imshow(cv2.cvtColor(cv2.imread(smiling_face), cv2.COLOR_BGR2GRAY), cmap='gray')
subplot(122)
imshow(cv2.cvtColor(cv2.imread(non_smiling_face), cv2.COLOR_BGR2GRAY), cmap='gray')

gray1, face1 = detect_face(cv2.imread(smiling_face))

gray2, face2 = detect_face(cv2.imread(non_smiling_face))

def test_recognition(c1, c2):
    subplot(121)
    extracted_face1 = extract_face_features(gray1, face1[0], (c1, c2))
    imshow(extracted_face1, cmap='gray')
    print(predict_face_is_smiling(extracted_face1).reshape(1, -1))
    subplot(122)
    extracted_face2 = extract_face_features(gray2, face2[0], (c1, c2))
    imshow(extracted_face2, cmap='gray')
    print(predict_face_is_smiling(extracted_face2))

widgets.interact(test_recognition,
         c1=(0.0, 0.3, 0.01),
         c2=(0.0, 0.3, 0.01))

extracted_faces = []
for facefile in [smiling_face, non_smiling_face]:
    gray, detected_faces = detect_face(cv2.imread(facefile))
    for face in detected_faces:
        extracted_face = extract_face_features(gray, face, offset_coefficients=(0.03, 0.05))
        extracted_faces.append(extracted_face)
        
imshow(mean(array(extracted_faces), axis=0), cmap='gray')


#################################################################################################################################
# Starts webcam that used OpenCV and new extracted features
#################################################################################################################################

cascPath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#################################################################################################################################
# Starts live camera, detects smile/no smile, press q to exit
#################################################################################################################################
def start_live_camera():
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detect_face(frame)
        
        face_index = 0
        
        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # extract features
                extracted_face = extract_face_features(gray, face, (0.03, 0.05)) #(0.075, 0.05)

                # predict smile
                prediction_result = predict_face_is_smiling(extracted_face)

                # draw extracted face in the top right corner
                frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

                # annotate main image with a label
                if prediction_result == 1:
                    cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                else:
                    cv2.putText(frame, "not smiling",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

                # increment counter
                face_index += 1
                    

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

#################################################################################################################################
# Define flask routes
#################################################################################################################################
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import requests, os, sys, json, time
from flask_cors import CORS

app = Flask(__name__, static_folder = "static/", template_folder = "templates/")
CORS(app)
	
@app.route('/detect', methods = ['POST'])
def detect_smile():
   start_live_camera()
   return redirect("http://localhost:8000/")

#################################################################################################################################
# Start web server using flask
#################################################################################################################################
if __name__ == '__main__':
   app.run(host='0.0.0.0')



