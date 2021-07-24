# Smile Detection
This project runs as a web app that has a Detect button that calls on my python script which runs as a web server. It sets up a video which uses your camera, and in turn OpenCV, to process your face and detect whether you are smiling or not. My model is 82% accurate, but when testing it out, it should just be your face in a well lit area with no distractions.

## Instructions and Notes
1. I have created some "dummy images", which you should replace with your own. In my code, face2.jpg and face3.jpg should be pictures of yourself, one in which you are smiling and one in which you are not smiling.
2. I have also created "dummy images" for the web server, smile.jpg and nosmile.jpg. These pictures can be whatever you want to decorate your web app. In my case, I took a picture of my face when the OpenCV was able to detect me smiling and not smiling. More details can be found on my github.io website.
3. I have already included code for the web server to start in the smileServer.py (Flask). To start the local host, do python -m http.server.

## Technologies Used 
- Python (Flask, Machine Learning)
- JavaScript 
- HTML
- CSS
- Bootstrap
- Jupyter Notebook
