# Smile Detection Model
I worked on a smile detection model using machine learning that can accurately predict when a person is smiling or not, given an image of the person. 

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Sanay B | Archbishop Mitty | Computer Science/Machine Learning | Incoming Junior

# Demo Night

Challenges: One of the main challenges I had was getting my Android App to work with my backend. Unfortunately, I was never able to successfully send an image to the backend through the app. My solution to create a web app that would have the same function. I was able to get it to send an image to the raspberry pi, and from there the rest of my code worked. Another challenge I had was manually uploading and annotating all my files onto Nanonets. My solution was to create a python script that would run a for loop through each of the images and do the annotations for me. 


Reflection: Overall, I realized I like the software/problem solving side of computer science a lot. Though debugging and errors can be really frustrating, fixing them and getting the code to work is really worth it. This project also helped me understand more about machine learning as I had never actually integrated it into my code before or really used it. The amount of paths and possibilies someone can explore with object detection cannot be numbered.


<iframe width="560" height="315" src="https://www.youtube.com/embed/Pa-UssOfi8E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Final Milestone
My final milestone is setting up my server on the raspberry pi instead of my own computer and calling the results from there, which it gets from Nanonets. The concept is the same as Milestone 2, except I was able to successfully integrate my raspberry pi into the project.


This is my code for setting up the web server, which would receive and save an image requested by the user on the frontend. The code would then call Nanonets, get a message back from it using its API, and send it back to the front end. The message will return something like "3. This is an Acura!"

![image](https://user-images.githubusercontent.com/60077919/124304073-13a1c900-db18-11eb-9613-484a7197fe0f.png)

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/hiAswzDwLWg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

# Second Milestone
My second milestone involved getting the web server set up on my computer. I created a Web App that would have a browse button that selects an image. When I submit and press get result, a file is saved in my directory, and I get a result appear on the frontend from the server that is talking to my model on the backend. My design is mostly my finalized.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/cn1QfGRkooc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

# First Milestone
  

My first milestone involved setting up my raspberry pi and being able to detect the car and its model with accuracy. I used the supervised training model Nanonets to formulate my car model, and I was able to create a python script that would upload images of the different car types I wanted to predict - Acura, Audi, Dodge, Ford, and Hyundai - and their corresponding annotations that I was able to get from a car dataset. This made it much easier to train the model as I did not have to draw bounding boxes manually. The code already did it for me. When my model was first created, it was at 88% accuracy, but after testing more and more car images, the accuracy kept increasing. Now, it is at 99.5%. I also created another python script, "prediction.py", that would be able to test images using the model solely through code and not the UI on Nanonets itself.


<center><iframe width="560" height="315" src="https://www.youtube.com/embed/QsjuReQxHeo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>
