# Smile Detection Model
I worked on a smile detection model using machine learning that can accurately predict when a person is smiling or not, given an image of the person. 

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Sanay B | Archbishop Mitty | Computer Science/Machine Learning | Incoming Junior

# Demo Night

Challenges: One of the main challenges I had was that some of the tutorials I was using had outdated code from 2015. For example, the cross_selection function that calls on KFold that splits the train and test data, was replaced by sklearn to model_selection. The two are the exact same, except for the name. Another major challenge I had was matching the image pixel size that was given by the olivetti faces dataset. The images in this dataset were 64 by 64, but images I got from google or took of myself were not always 64 by 64. As a result, I used the code below to extract a 64 by 64 image of the image I had already inputted, by matching coefficients for each image and multiplying it by the height and width to get that exact pixel size.


![image](https://user-images.githubusercontent.com/60077919/126688915-27157415-da6c-497e-9f93-66d65fa377c5.png)


Reflection: Overall, I realized I like the software/problem solving side of computer science a lot. Though debugging and errors can be really frustrating, fixing them and getting the code to work is really worth it. This project also helped me understand more about machine learning as I had never actually integrated it into my code before or really used it. The amount of paths and possibilies someone can explore with OpenCV and image classification cannot be numbered.


<iframe width="560" height="315" src="https://www.youtube.com/embed/Pa-UssOfi8E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Final Milestone
My final milestone is setting up my server on the raspberry pi instead of my own computer and calling the results from there, which it gets from Nanonets. The concept is the same as Milestone 2, except I was able to successfully integrate my raspberry pi into the project.


This is my code for setting up the web server, which would receive and save an image requested by the user on the frontend. The code would then call Nanonets, get a message back from it using its API, and send it back to the front end. The message will return something like "3. This is an Acura!"

![image](https://user-images.githubusercontent.com/60077919/124304073-13a1c900-db18-11eb-9613-484a7197fe0f.png)

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/hiAswzDwLWg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

# Second Milestone
The first part of my second milestone involved getting my model to actually work and detect faces when I upload them. It was decently accurate. My model, once trained and tested, had an accuracy of 82%. After this was done, I used this model to be able to detect when I am smiling or not using a live webcam. This part worked much better as I had provided coefficients for my face that made it easier for the model algorithm to correctly identify whether I was smiling or not.


![image](https://user-images.githubusercontent.com/60077919/126688752-bf4b0f21-8fc4-4452-9ecf-4b3468865e87.png)


<center><iframe width="560" height="315" src="https://www.youtube.com/embed/a5R4cG8g_pQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

# First Milestone
  

My first milestone involved setting up my raspberry pi and being able to display and classify the olivetti faces dataset using Jupyter Notebook. My code came up with a simple UI that had two buttons, smile and sad face, which I used to classify 200 unique images. Then, I was able to dump the data into an xml file, load it, and classify each one of them as smiling or not smiling using a string. Based on the true and false values from when I first pressed the buttons, my code was able to match that to whether the images were smiling or not. True means that they are, and false means that they are not.


<center><iframe width="560" height="315" src="https://www.youtube.com/embed/RxNrnyGkhDE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>
